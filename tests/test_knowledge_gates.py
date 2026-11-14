"""Unit tests for knowledge base quality gates, record/table chunking, freshness filtering, and activation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from agent.config import Settings
from rag.calibrate import calibrate_threshold, run_calibration
from rag.calibrate import main as calibrate_main
from rag.chunking import (
    chunk_json_node,
    chunk_text_and_tables,
    derive_dates_and_topic,
    is_table_block,
)
from rag.coverage import approve_coverage, compute_sha256, generate_coverage_report
from rag.coverage import main as coverage_main
from rag.eval import (
    check_evidence_hit,
    compute_mrr,
    compute_percentile,
    normalize_text,
    run_evaluation,
)
from rag.eval import (
    main as eval_main,
)
from rag.ingest import activate, rollback
from rag.manifest import read_active
from rag.retriever import BM25Index, Evidence, filter_freshness, rrf


class TestRecordAndTableChunking(unittest.TestCase):
    def test_json_nested_context_propagation(self) -> None:
        """Department -> Faculty -> Publications keeps parent identifying context in every child."""
        nested_data = {
            "name": "School of Engineering",
            "code": "SEAS",
            "departments": [
                {
                    "name": "Computer Science and Engineering",
                    "code": "CSE",
                    "faculty": [
                        {
                            "name": "Dr. Ramesh Kumar",
                            "designation": "Associate Professor",
                            "research_areas": ["Distributed Systems", "Cloud Computing"],
                            "publications": [
                                {
                                    "title": "Fault-Tolerant Consensus in Edge Networks",
                                    "year": 2024,
                                    "venue": "IEEE Cloud",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        chunks = chunk_json_node(nested_data)
        self.assertGreater(len(chunks), 1)

        # Check that publication chunk contains parent CSE and Dr. Ramesh Kumar context
        pub_chunks = [c for c in chunks if "Fault-Tolerant Consensus" in c[2]]
        self.assertEqual(len(pub_chunks), 1)
        loc, ctx, text, scalars = pub_chunks[0]
        self.assertIn("CSE", ctx)
        self.assertIn("Dr. Ramesh Kumar", ctx)
        self.assertIn("IEEE Cloud", text)

    def test_table_row_protection_and_header_repeat(self) -> None:
        """Fee table keeps rows intact and repeats header when chunk splits."""
        header = "Program Name\tTuition Fee (Per Year)\tHostel Fee"
        rows = [f"B.Tech CSE Specialization {i}\tINR 3,50,000\tINR 1,20,000" for i in range(1, 25)]
        table_text = "\n".join([header] + rows)

        chunks = chunk_text_and_tables(table_text, page=1, max_chars=500)
        self.assertGreater(len(chunks), 1)
        for loc, chunk_text in chunks:
            # Header must repeat at top of every chunk cut from the same table
            lines = chunk_text.splitlines()
            self.assertEqual(lines[0], header)
            # Row must not be split across chunks
            for row in lines[1:]:
                self.assertIn("INR 3,50,000", row)

    def test_is_table_block_detection(self) -> None:
        lines = [
            "Col1    Col2    Col3",
            "Val1    Val2    Val3",
            "Val4    Val5    Val6",
        ]
        self.assertTrue(is_table_block(lines))

        prose_lines = [
            "This is a regular sentence that goes on without multiple columns.",
            "Another sentence describing university policy rules.",
        ]
        self.assertFalse(is_table_block(prose_lines))


class TestDateDerivationAndFreshness(unittest.TestCase):
    def test_date_derivation_order(self) -> None:
        """Derivation priority: record -> sidecar -> filename -> null."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            doc_file = tmp / "hostel_fees__2025-07-01.pdf"
            doc_file.touch()

            # Filename convention
            eff, val, ver, top = derive_dates_and_topic(doc_file)
            self.assertEqual(eff, "2025-07-01")
            self.assertEqual(top, "hostel_fees")

            # Sidecar takes precedence over filename
            sidecar = tmp / "hostel_fees__2025-07-01.pdf.meta.yaml"
            sidecar.write_text("effective_date: '2026-08-01'\ntopic_key: 'hostel_fees_v2'\n", encoding="utf-8")
            eff2, val2, ver2, top2 = derive_dates_and_topic(doc_file)
            self.assertEqual(eff2, "2026-08-01")
            self.assertEqual(top2, "hostel_fees_v2")

            # Record scalars take precedence over sidecar
            eff3, val3, ver3, top3 = derive_dates_and_topic(doc_file, record={"effective_date": "2027-01-01"})
            self.assertEqual(eff3, "2027-01-01")

    def test_freshness_filtering_latest_effective_date(self) -> None:
        """Two documents for different years return only latest effective one."""
        today_str = "2026-11-14"
        candidates = [
            (("c1", "Fee 2024 is 2.5L", {"topic_key": "fees", "effective_date": "2024-07-01"}), 0.8),
            (("c2", "Fee 2025 is 3.0L", {"topic_key": "fees", "effective_date": "2025-07-01"}), 0.85),
            (("c3", "Fee 2026 is 3.5L", {"topic_key": "fees", "effective_date": "2026-07-01"}), 0.9),
            (("c4", "Fee 2027 is 4.0L", {"topic_key": "fees", "effective_date": "2027-07-01"}), 0.95),  # future!
        ]

        filtered, note = filter_freshness(candidates, today_str)
        self.assertEqual(note, "")
        # Latest effective <= today is 2026-07-01 (c3)
        chunk_ids = [c[0][0] for c in filtered]
        self.assertIn("c3", chunk_ids)
        self.assertNotIn("c1", chunk_ids)
        self.assertNotIn("c2", chunk_ids)
        self.assertNotIn("c4", chunk_ids)

    def test_freshness_expired_only_produces_note(self) -> None:
        """When only expired chunks remain, prefix with NOTE and keep them."""
        today_str = "2026-11-14"
        candidates = [
            (("c_old", "Old rule text", {"valid_until": "2024-01-01", "effective_date": "2023-01-01"}), 0.8),
        ]
        filtered, note = filter_freshness(candidates, today_str)
        self.assertEqual(len(filtered), 1)
        self.assertIn("NOTE: this information may be out of date", note)
        self.assertIn("2023-01-01", note)

    def test_deterministic_tie_breaking(self) -> None:
        """Ties break by score desc, effective_date desc, doc_version desc, chunk_id asc."""
        today_str = "2026-11-14"
        candidates = [
            (("id_b", "text b", {"effective_date": "2025-01-01", "doc_version": "1.0"}), 0.8),
            (("id_a", "text a", {"effective_date": "2025-01-01", "doc_version": "1.0"}), 0.8),
        ]
        filtered, _ = filter_freshness(candidates, today_str)
        # Same score, date, version -> alphabetical chunk_id asc: id_a before id_b
        self.assertEqual(filtered[0][0][0], "id_a")
        self.assertEqual(filtered[1][0][0], "id_b")


class TestCalibrationAndSplitHygiene(unittest.TestCase):
    def test_split_leakage_rejection(self) -> None:
        """Calibration strictly refuses any item whose split is 'heldout'."""
        leak_items = [
            {"id": "Q1", "split": "calibration", "question": "q1", "category": "answerable"},
            {"id": "Q2_LEAK", "split": "heldout", "question": "q2", "category": "answerable"},
        ]
        with self.assertRaises(ValueError) as ctx:
            calibrate_threshold(leak_items, score_func=lambda q, item: 0.8)
        self.assertIn("Split leakage", str(ctx.exception))
        self.assertIn("Q2_LEAK", str(ctx.exception))

    def test_percentile_calculation_known_values(self) -> None:
        """Percentile function math test on known array."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(compute_percentile(data, 0), 10.0)
        self.assertEqual(compute_percentile(data, 50), 30.0)
        self.assertEqual(compute_percentile(data, 100), 50.0)

    def test_evidence_hit_logic(self) -> None:
        """Evidence hit requires correct source (and page) and normalized must_contain."""
        ev = [
            Evidence(
                chunk_id="c1",
                text="The annual tuition fee for B.Tech is INR 3,50,000.",
                source="pdf/Fee_Structure.pdf",
                page_or_path="page 4",
                score=0.9,
            )
        ]
        # Match
        rule_match = [{"source": "Fee_Structure.pdf", "page": 4, "must_contain": "3,50,000"}]
        self.assertTrue(check_evidence_hit(ev, rule_match))

        # Page mismatch
        rule_wrong_page = [{"source": "Fee_Structure.pdf", "page": 5, "must_contain": "3,50,000"}]
        self.assertFalse(check_evidence_hit(ev, rule_wrong_page))

        # Fact mismatch
        rule_wrong_fact = [{"source": "Fee_Structure.pdf", "page": 4, "must_contain": "5,00,000"}]
        self.assertFalse(check_evidence_hit(ev, rule_wrong_fact))

    def test_mrr_and_text_normalization(self) -> None:
        """MRR score computation and text normalization."""
        ev = [Evidence(chunk_id="c1", text="Sample text", source="s1", page_or_path="1", score=0.8)]
        mrr = compute_mrr(ev, [{"source": "s1", "must_contain": "sample"}])
        self.assertEqual(mrr, 1.0)
        self.assertEqual(normalize_text("Hello, World! 123"), "hello world 123")


class TestCoverageReportingAndApproval(unittest.TestCase):
    def test_generate_report_and_approve(self) -> None:
        """Coverage report generation and binding approval hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest = {
                "version": "test_v",
                "chunks": 50,
                "sources": [
                    {"source": "pdf/Hostel-Policy.pdf", "chunks": 15, "empty_pages": [], "effective_date": "2025-07-01"},
                ],
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = generate_coverage_report(tmp)
            self.assertIn("total_chunks", report)
            self.assertTrue((tmp / "coverage_report.json").is_file())

            approval = approve_coverage(tmp, approved_by="Compliance Officer")
            self.assertEqual(approval["approved_by"], "Compliance Officer")
            self.assertEqual(len(approval["report_sha256"]), 64)
            self.assertEqual(approval["report_sha256"], compute_sha256((tmp / "coverage_report.json").read_bytes()))


class TestKnowledgeLifecycleAndGates(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.settings = Settings(chroma_dir=self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_artifact(
        self,
        version: str,
        quality_status: str = "passed",
        partial: bool = False,
        in_staging: bool = True,
        approved_by: str | None = "Audit Lead",
    ) -> Path:
        base = self.root / "staging" / version if in_staging else self.root / version
        base.mkdir(parents=True, exist_ok=True)
        (base / "chroma.sqlite3").touch()

        # Write coverage report
        cov_report = {"partial": partial, "total_chunks": 100}
        cov_report_file = base / "coverage_report.json"
        cov_report_file.write_text(json.dumps(cov_report), encoding="utf-8")
        cov_hash = hashlib.sha256(cov_report_file.read_bytes()).hexdigest()

        approval = None
        if approved_by:
            approval = {
                "approved_by": approved_by,
                "approved_at_utc": "2026-11-14T00:00:00Z",
                "report_sha256": cov_hash,
            }

        manifest = {
            "schema_version": 1,
            "version": version,
            "collection": "nexi",
            "embedding": {
                "embed_model": self.settings.embed_model,
                "embed_revision": self.settings.embed_revision,
                "normalize_embeddings": True,
                "distance": "cosine",
            },
            "chunks": 100,
            "partial": partial,
            "quality_status": quality_status,
            "coverage_approval": approval,
            "coverage_report": "coverage_report.json",
        }
        (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return base

    def test_activation_refuses_invalid_states(self) -> None:
        """Activation refuses when partial, unevaluated, failed, needs_review, or approval hash mismatched."""
        v = "11111111111111111111111111111111"

        # 1. Partial
        self._create_mock_artifact(v, quality_status="passed", partial=True)
        with self.assertRaises(RuntimeError) as ctx:
            activate(self.settings, v)
        self.assertIn("Cannot activate partial", str(ctx.exception))

        # 2. Unevaluated
        self._create_mock_artifact(v, quality_status="unevaluated", partial=False)
        with self.assertRaises(RuntimeError) as ctx:
            activate(self.settings, v)
        self.assertIn("must be 'passed'", str(ctx.exception))

        # 3. Needs review
        self._create_mock_artifact(v, quality_status="needs_review", partial=False)
        with self.assertRaises(RuntimeError) as ctx:
            activate(self.settings, v)
        self.assertIn("must be 'passed'", str(ctx.exception))

        # 4. Hash mismatch (coverage report modified after approval)
        base = self._create_mock_artifact(v, quality_status="passed", partial=False)
        (base / "coverage_report.json").write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(RuntimeError) as ctx:
            activate(self.settings, v)
        self.assertIn("hash mismatch", str(ctx.exception))

    def test_atomic_activation_and_rollback(self) -> None:
        """Atomic swap works, updates ACTIVE and CURRENT, and rollback restores previous."""
        v1 = "11111111111111111111111111111111"
        v2 = "22222222222222222222222222222222"

        self._create_mock_artifact(v1, quality_status="passed", partial=False)
        activate(self.settings, v1)
        self.assertEqual((self.root / "ACTIVE").read_text(encoding="utf-8").strip(), v1)
        self.assertEqual((self.root / "CURRENT").read_text(encoding="utf-8").strip(), v1)

        self._create_mock_artifact(v2, quality_status="passed", partial=False)
        activate(self.settings, v2)
        self.assertEqual((self.root / "ACTIVE").read_text(encoding="utf-8").strip(), v2)

        # Rollback
        rolled = rollback(self.settings)
        self.assertEqual(rolled, v1)
        self.assertEqual((self.root / "ACTIVE").read_text(encoding="utf-8").strip(), v1)

    def test_last_three_retention_pruning(self) -> None:
        """Pruning retains the last 3 versions plus active, deleting older ones."""
        versions = [f"{i:032d}" for i in range(1, 6)]
        for v in versions:
            self._create_mock_artifact(v, quality_status="passed", partial=False)
            activate(self.settings, v)

        # Versions 1 and 2 were pruned during activation; 3, 4, 5 kept
        self.assertFalse((self.root / "artifacts" / versions[0]).exists())
        self.assertFalse((self.root / "artifacts" / versions[1]).exists())
        self.assertTrue((self.root / "artifacts" / versions[2]).exists())
        self.assertTrue((self.root / "artifacts" / versions[3]).exists())
        self.assertTrue((self.root / "artifacts" / versions[4]).exists())

    def test_runtime_refuses_unverified_unless_override(self) -> None:
        """Runtime refuses unverified artifact unless ALLOW_UNVERIFIED_KNOWLEDGE=1."""
        v = "33333333333333333333333333333333"
        self._create_mock_artifact(v, quality_status="phase1_unevaluated", partial=True, in_staging=False)
        (self.root / "ACTIVE").write_text(v + "\n", encoding="utf-8")

        # Default settings: refuse!
        with self.assertRaises(RuntimeError) as ctx:
            read_active(self.settings)
        self.assertIn("unverified", str(ctx.exception))

        # Dev override settings: allows with warning
        settings_override = Settings(chroma_dir=self.root, allow_unverified_knowledge=True)
        path, manifest = read_active(settings_override)
        self.assertEqual(manifest["version"], v)

    def test_independent_gate_failures_and_latency_waiver(self) -> None:
        """Each gate can fail independently; latency waiver only waives latency."""
        v = "44444444444444444444444444444444"
        base = self._create_mock_artifact(v, quality_status="unevaluated", partial=False)

        mock_retriever = MagicMock()
        # Mock search result with zero evidence
        mock_res = MagicMock()
        mock_res.evidence = ()
        mock_retriever.search.return_value = mock_res

        # Test with custom gates yaml in temp
        gates_yaml = self.root / "gates.yaml"
        gates_yaml.write_text(
            "gates:\n  heldout_evidence_hit_at_4: 0.90\n  retrieval_p95_ms: 10.0\n",
            encoding="utf-8",
        )
        questions_yaml = self.root / "questions.yaml"
        questions_yaml.write_text(
            "questions:\n  - id: Q1\n    split: heldout\n    category: answerable\n    question: q1\n    expected_evidence: [{source: s1, must_contain: fact}]\n",
            encoding="utf-8",
        )

        # 1. Fails accuracy gate -> status failed
        report, passed = run_evaluation(
            artifact_dir=base,
            settings=self.settings,
            questions_path=questions_yaml,
            gates_path=gates_yaml,
            mock_retriever=mock_retriever,
        )
        self.assertEqual(report["quality_status"], "failed")
        self.assertFalse(passed)

        # 2. Mock hit passing, but latency exceeding 10ms
        mock_hit_res = MagicMock()
        mock_hit_res.evidence = (
            Evidence(chunk_id="c1", text="fact is here", source="s1", page_or_path="1", score=0.9),
        )
        mock_retriever.search.return_value = mock_hit_res

        # Without waiver: fails on latency!
        with patch("rag.eval.perf_counter", side_effect=[0.0, 0.05, 0.0, 0.05, 0.0, 0.05]):
            report2, passed2 = run_evaluation(
                artifact_dir=base,
                settings=self.settings,
                questions_path=questions_yaml,
                gates_path=gates_yaml,
                mock_retriever=mock_retriever,
            )
            self.assertEqual(report2["quality_status"], "failed")

        # With waiver: passes!
        with patch("rag.eval.perf_counter", side_effect=[0.0, 0.05, 0.0, 0.05, 0.0, 0.05]):
            report3, passed3 = run_evaluation(
                artifact_dir=base,
                settings=self.settings,
                questions_path=questions_yaml,
                gates_path=gates_yaml,
                mock_retriever=mock_retriever,
                latency_waiver_reason="Known CPU emulation slow path",
                latency_waiver_approver="VP Engineering",
            )
            self.assertTrue(passed3)
            self.assertEqual(report3["quality_status"], "needs_review")  # items have reviewed_by: null
            self.assertIsNotNone(report3["latency_waiver"])


class TestBM25AndFusion(unittest.TestCase):
    def test_bm25_and_rrf(self) -> None:
        corpus = [
            "SRM University AP hostel fee structure and refund rules",
            "Computer Science engineering faculty and research lab",
            "Student attendance criteria and leave application process",
        ]
        bm25 = BM25Index(corpus)
        # Query matching document 0
        hits = bm25.top_k("hostel fee", k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0], 0)

        # Query matching document 2
        hits2 = bm25.top_k("attendance leave policy", k=2)
        self.assertEqual(hits2[0], 2)

        # Empty or unseen terms
        self.assertEqual(bm25.top_k(""), [])
        self.assertEqual(bm25.top_k("xyz123 nonexistent query"), [])

        # RRF fusion
        dense_ids = ["doc_a", "doc_b", "doc_c"]
        bm25_ids = ["doc_b", "doc_d", "doc_a"]
        fused = rrf(dense_ids, bm25_ids, k=60)
        # doc_b is rank 2 in dense and rank 1 in bm25, should be #1 overall
        self.assertEqual(fused[0], "doc_b")
        self.assertIn("doc_a", fused)
        self.assertIn("doc_c", fused)
        self.assertIn("doc_d", fused)


class TestCalibrationAndClis(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.settings = Settings(chroma_dir=self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calibration_edge_cases(self) -> None:
        # 1. No answerable items
        unans_items = [
            {"id": "U1", "split": "calibration", "category": "out_of_domain", "abstain": True}
        ]
        thresh1, _ = calibrate_threshold(unans_items, lambda q, it: 0.1)
        self.assertEqual(thresh1, 0.20)

        # 2. No unanswerable items
        ans_items = [
            {"id": "A1", "split": "calibration", "category": "answerable", "abstain": False}
        ]
        thresh2, _ = calibrate_threshold(ans_items, lambda q, it: 0.5)
        self.assertAlmostEqual(thresh2, 0.40)

        # 3. Split leakage detection
        leaked = [{"id": "L1", "split": "heldout", "category": "answerable"}]
        with self.assertRaises(ValueError):
            calibrate_threshold(leaked, lambda q, it: 0.5)

    def test_run_calibration_and_cli(self) -> None:
        import sys

        artifact_dir = self.root / ("33" * 16)
        artifact_dir.mkdir(parents=True)
        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"version": artifact_dir.name, "chunks": 1}))

        q_path = self.root / "questions.yaml"
        q_path.write_text(
            yaml.dump({
                "questions": [
                    {"id": "C1", "split": "calibration", "category": "answerable", "question": "q1"},
                    {"id": "C2", "split": "calibration", "category": "out_of_domain", "question": "q2", "abstain": True},
                ]
            })
        )

        def mock_score(q: str, it: dict) -> float:
            return 0.85 if it.get("category") == "answerable" else 0.15

        stats = run_calibration(
            artifact_dir=artifact_dir,
            settings=self.settings,
            questions_path=q_path,
            mock_score_func=mock_score,
        )
        self.assertAlmostEqual(stats["chosen_threshold"], 0.50)
        self.assertEqual(stats["items_used"], 2)

        # Test CLI main
        with patch.object(sys, "argv", ["calibrate.py", artifact_dir.name, "--questions", str(q_path)]), \
             patch("rag.calibrate.load_settings", return_value=self.settings), \
             patch("rag.calibrate.run_calibration", return_value=stats):
            calibrate_main()

    def test_eval_cli_and_coverage_cli(self) -> None:
        import sys

        artifact_dir = self.root / ("44" * 16)
        artifact_dir.mkdir(parents=True)
        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"version": artifact_dir.name, "chunks": 1, "sources": []}))

        # Test coverage CLI
        with patch.object(sys, "argv", ["coverage.py", "report", artifact_dir.name]), \
             patch("rag.coverage.load_settings", return_value=self.settings), \
             patch("rag.coverage.generate_coverage_report", return_value={"status": "ok", "partial": False}):
            coverage_main()

        # Test eval CLI
        with patch.object(sys, "argv", ["eval.py", artifact_dir.name]), \
             patch("rag.eval.load_settings", return_value=self.settings), \
             patch("rag.eval.run_evaluation", return_value=({"metrics": {}, "quality_status": "evaluated"}, True)):
            eval_main()


class TestRetrieverSyncAndAsync(unittest.IsolatedAsyncioTestCase):
    async def test_retriever_pipeline(self) -> None:
        import numpy as np

        from rag.manifest import COLLECTION, SCHEMA_VERSION, model_fingerprint
        from rag.retriever import Retriever

        temp_dir = tempfile.mkdtemp()
        root = Path(temp_dir)
        artifact_dir = root / ("55" * 16)
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "chroma.sqlite3").write_text("dummy")

        settings = Settings(chroma_dir=root, allow_unverified_knowledge=True)

        manifest_data = {
            "schema_version": SCHEMA_VERSION,
            "version": artifact_dir.name,
            "collection": COLLECTION,
            "embedding": model_fingerprint(settings),
            "chunks": 2,
            "rerank_min_score": 0.40,
            "sources": [],
            "quality_status": "evaluated",
            "partial": False,
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest_data))
        (root / "ACTIVE").write_text(artifact_dir.name)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["Hostel fee is 50000 rupees", "Library timings are 9am to 9pm"],
            "metadatas": [
                {"source": "hostel.pdf", "page": "1", "topic_key": "hostel"},
                {"source": "library.pdf", "page": "2", "topic_key": "library"},
            ],
        }
        mock_collection.query.return_value = {
            "ids": [["c1", "c2"]],
            "documents": [["Hostel fee is 50000 rupees", "Library timings are 9am to 9pm"]],
            "metadatas": [[{"source": "hostel.pdf", "page": "1"}, {"source": "library.pdf", "page": "2"}]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([0.1, 0.2, 0.3])

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.85, 0.20])

        with patch("chromadb.PersistentClient", return_value=mock_client), \
             patch("sentence_transformers.SentenceTransformer", return_value=mock_st), \
             patch("sentence_transformers.CrossEncoder", return_value=mock_ce):
            retriever = Retriever(settings)
            self.assertEqual(retriever.knowledge_version, artifact_dir.name)

            # 1. Embedding cache test
            vec1 = retriever._get_query_embedding("hostel fee")
            vec2 = retriever._get_query_embedding("hostel fee")
            self.assertEqual(vec1, vec2)
            # st.encode should be called once due to cache
            self.assertEqual(mock_st.encode.call_count, 1)

            # 2. Search async
            res = await retriever.search("hostel fee")
            self.assertEqual(len(res.evidence), 1)
            self.assertEqual(res.evidence[0].chunk_id, "c1")
            self.assertEqual(res.evidence[0].score, 0.85)

            # 3. Below threshold test
            mock_ce.predict.return_value = np.array([0.10, 0.15])
            res_below = await retriever.search("random query")
            self.assertEqual(len(res_below.evidence), 0)
            self.assertEqual(res_below.reason, "below_threshold")

            retriever.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
