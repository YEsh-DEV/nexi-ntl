"""In-memory sliding-window rate limiter with LRU eviction and trusted proxy parsing."""

from __future__ import annotations

import ipaddress
from collections import OrderedDict
from time import monotonic
from typing import Callable

from agent.config import Settings


def parse_trusted_proxies(
    proxies: tuple[str, ...] | list[str],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for p in proxies:
        p_clean = p.strip()
        if not p_clean:
            continue
        try:
            if "/" in p_clean:
                networks.append(ipaddress.ip_network(p_clean, strict=False))
            else:
                networks.append(
                    ipaddress.ip_network(
                        f"{p_clean}/32" if ":" not in p_clean else f"{p_clean}/128"
                    )
                )
        except ValueError:
            pass
    return networks


def is_ip_in_trusted_proxies(
    ip_str: str,
    trusted_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        return any(ip in net for net in trusted_networks)
    except ValueError:
        return False


def resolve_client_ip(
    peer_ip: str,
    x_forwarded_for: str | None,
    trusted_proxies: tuple[str, ...] | list[str],
) -> str:
    """Resolve real client IP.

    Uses socket peer address by default. Uses X-Forwarded-For only when TRUSTED_PROXIES
    is set and peer_ip is in it, taking the right-most address that is NOT a trusted proxy.
    Never trusts the header otherwise.
    """
    clean_peer = peer_ip.strip()
    if not trusted_proxies or not x_forwarded_for:
        return clean_peer

    trusted_nets = parse_trusted_proxies(trusted_proxies)
    if not is_ip_in_trusted_proxies(clean_peer, trusted_nets):
        # Socket peer is not in trusted proxies; never trust header
        return clean_peer

    # Parse X-Forwarded-For chain from right to left
    parts = [p.strip() for p in x_forwarded_for.split(",") if p.strip()]
    for candidate in reversed(parts):
        if not is_ip_in_trusted_proxies(candidate, trusted_nets):
            return candidate

    return clean_peer


class RateLimiter:
    """Sliding-window rate limiter for client IP and global request volume."""

    def __init__(
        self,
        settings: Settings,
        clock: Callable[[], float] | None = None,
        max_clients: int = 10000,
    ) -> None:
        self.settings = settings
        self.clock = clock or monotonic
        self.max_clients = max_clients
        # OrderedDict for LRU eviction of client IP states
        self._client_windows: OrderedDict[str, list[float]] = OrderedDict()
        self._global_window: list[float] = []

    def check_rate_limit(
        self,
        peer_ip: str,
        x_forwarded_for: str | None = None,
    ) -> tuple[bool, int, str]:
        """Check if request is allowed.

        Returns:
            (allowed: bool, retry_after_s: int, client_ip: str)
        """
        now = self.clock()
        client_ip = resolve_client_ip(peer_ip, x_forwarded_for, self.settings.trusted_proxies)

        # 1. Check global rate limit (sliding 60-second window)
        global_window_s = 60.0
        self._global_window = [t for t in self._global_window if (now - t) < global_window_s]
        if len(self._global_window) >= self.settings.token_rate_global_per_min:
            oldest = self._global_window[0]
            retry_after = max(1, int(global_window_s - (now - oldest) + 0.999))
            return False, retry_after, client_ip

        # 2. Check per-client rate limit (sliding token_rate_window_s window)
        ip_window_s = self.settings.token_rate_window_s
        limit_per_ip = self.settings.token_rate_per_ip

        if client_ip in self._client_windows:
            self._client_windows.move_to_end(client_ip)
            history = self._client_windows[client_ip]
        else:
            history = []
            if len(self._client_windows) >= self.max_clients:
                self._client_windows.popitem(last=False)  # Evict oldest (LRU)
            self._client_windows[client_ip] = history

        history = [t for t in history if (now - t) < ip_window_s]
        self._client_windows[client_ip] = history

        if len(history) >= limit_per_ip:
            oldest = history[0]
            retry_after = max(1, int(ip_window_s - (now - oldest) + 0.999))
            return False, retry_after, client_ip

        # Allowed: record timestamps
        history.append(now)
        self._global_window.append(now)
        return True, 0, client_ip
