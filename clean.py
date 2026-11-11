"""Compatibility launcher; prefer python -m agent.main <command>."""

from agent.config import ConfigurationError
from agent.main import main

if __name__ == "__main__":
    try:
        main()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from None
