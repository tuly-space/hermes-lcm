"""Profile-scoped view of ``LCM_*`` and provider-key environment settings.

Hermes can import one copy of this plugin per profile into a single process (the
multiplexing gateway, and the cron ticker of any gateway). ``os.environ`` then holds
only the *launch* profile's ``.env``; reading it from another profile's copy binds
that copy to the launch profile's database and settings. Two copies sharing one
``lcm.db`` inside one process defeat SQLite's POSIX locks (closing either copy's
descriptor cancels both copies' locks) and corrupt the file.

Every LCM setting therefore resolves against the plugin copy's *own* profile home,
captured once at registration:

1. ``<home>/.env`` (the profile's own file, parsed without mutating ``os.environ``);
2. ``os.environ`` — only when that home is the process home (single-profile gateway,
   CLI, tests), so systemd ``Environment=`` and ``op run`` injection keep working.

A foreign profile's copy never falls back to ``os.environ``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional

_LOCK = threading.RLock()
_bound_home: Optional[Path] = None
_file_cache: dict[str, tuple[tuple[int, int, int], dict[str, str]]] = {}


def _resolve(path: str | os.PathLike[str]) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path).expanduser()


def _process_home() -> Optional[Path]:
    """The process launch home: ``HERMES_HOME`` (set by every service unit), else ``~/.hermes``.

    Deliberately not the context-local override: that override is how the host
    scopes a secondary profile, and the process home must not follow it.
    """
    raw = os.environ.get("HERMES_HOME", "").strip()
    return _resolve(raw) if raw else _resolve(Path.home() / ".hermes")


def _active_home() -> Optional[Path]:
    try:
        from hermes_cli.config import get_hermes_home  # type: ignore

        return _resolve(get_hermes_home())
    except Exception:
        raw = os.environ.get("HERMES_HOME", "").strip()
        return _resolve(raw) if raw else None


def bind_home(home: str | os.PathLike[str] | None) -> Optional[Path]:
    """Pin this plugin copy to ``home`` (called from ``register``). Idempotent."""
    global _bound_home
    with _LOCK:
        _bound_home = _resolve(home) if home else _active_home()
        return _bound_home


def bound_home() -> Optional[Path]:
    with _LOCK:
        return _bound_home if _bound_home is not None else _active_home()


def is_process_home(home: Optional[Path] = None) -> bool:
    home = home if home is not None else bound_home()
    proc = _process_home()
    return home is None or proc is None or home == proc


def _strip_inline_comment(value: str) -> str:
    in_single = in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        return inner
    return _strip_inline_comment(value)


def _parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("_", "").isalnum():
            continue
        values[key] = _parse_value(value)
    return values


def _load_env_file(home: Path) -> dict[str, str]:
    """Parse ``<home>/.env``; prefer Hermes' own parser so quoting rules match the host."""
    env_path = home / ".env"
    try:
        from agent.secret_scope import load_env_file  # type: ignore

        return dict(load_env_file(env_path))
    except Exception:
        pass
    try:
        st = env_path.stat()
    except OSError:
        return {}
    signature = (st.st_ino, st.st_size, st.st_mtime_ns)
    key = str(env_path)
    with _LOCK:
        cached = _file_cache.get(key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    values = _parse_env_text(text)
    with _LOCK:
        _file_cache[key] = (signature, dict(values))
    return values


class ProfileEnviron(Mapping[str, str]):
    """Read-only ``os.environ`` stand-in scoped to the plugin copy's profile home."""

    def _layers(self) -> tuple[dict[str, str], bool]:
        home = bound_home()
        file_values = _load_env_file(home) if home is not None else {}
        return file_values, is_process_home(home)

    def get(self, key: str, default=None):  # type: ignore[override]
        file_values, own_process = self._layers()
        if key in file_values:
            return file_values[key]
        if own_process:
            return os.environ.get(key, default)
        return default

    def __getitem__(self, key: str) -> str:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        file_values, own_process = self._layers()
        return key in file_values or (own_process and key in os.environ)

    def _keys(self) -> set[str]:
        file_values, own_process = self._layers()
        keys = set(file_values)
        if own_process:
            keys.update(os.environ)
        return keys

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._keys()))

    def __len__(self) -> int:
        return len(self._keys())


environ = ProfileEnviron()


def getenv(key: str, default=None):
    return environ.get(key, default)


def reset_for_tests() -> None:
    global _bound_home
    with _LOCK:
        _bound_home = None
        _file_cache.clear()
