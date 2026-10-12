"""Process-wide owner registry for LCM SQLite databases.

Hermes may import several copies of this plugin into one process: one per profile
home, as ``hermes_plugins.hermes_lcm`` and ``hermes_plugins.hermes_lcm__home_<digest>``.
Each copy has its own module globals, so the registry lives on the shared ``sys``
module. Engines (and their clones) of the SAME profile share a database normally.
A copy bound to a DIFFERENT profile home must never open the same inode: POSIX
locks are per process, and the second copy's descriptors silently disable the
first copy's locking.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_ATTR = "_hermes_lcm_db_owners_v1"


def _registry() -> tuple[threading.Lock, dict[tuple[int, int], dict]]:
    state = getattr(sys, _ATTR, None)
    if state is None:
        state = (threading.Lock(), {})
        setattr(sys, _ATTR, state)
    return state


class CrossProfileDatabaseError(RuntimeError):
    """Raised when a second profile's LCM copy tries to open a database in use."""


def _identity(db_path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(db_path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def acquire(db_path: Path, owner_home: str) -> tuple[int, int] | None:
    """Register ``owner_home`` as the user of ``db_path``; refuse a foreign home."""
    ident = _identity(db_path)
    if ident is None:
        return None
    owner = str(owner_home or "")
    lock, owners = _registry()
    with lock:
        entry = owners.get(ident)
        if entry is not None and entry["home"] != owner:
            raise CrossProfileDatabaseError(
                f"LCM database {db_path} is already open in this process for Hermes home "
                f"{entry['home']!r}; refusing a second copy bound to {owner!r}. Two plugin "
                "copies on one SQLite file cancel each other's POSIX locks and corrupt it."
            )
        if entry is None:
            owners[ident] = {"home": owner, "count": 1}
        else:
            entry["count"] += 1
    return ident


def release(ident: tuple[int, int] | None) -> None:
    if ident is None:
        return
    lock, owners = _registry()
    with lock:
        entry = owners.get(ident)
        if entry is None:
            return
        entry["count"] -= 1
        if entry["count"] <= 0:
            owners.pop(ident, None)


def snapshot() -> dict[tuple[int, int], dict]:
    lock, owners = _registry()
    with lock:
        return {k: dict(v) for k, v in owners.items()}
