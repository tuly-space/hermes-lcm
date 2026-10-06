"""Two profiles' copies of hermes-lcm loaded into ONE process (multiplexing gateway).

Regression for lcm.db corruption: a second profile's copy read the launch profile's
LCM_DATABASE_PATH from os.environ, opened the same file, and its descriptor close
cancelled the first copy's POSIX locks.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _run(script: str, env: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess:
    path = tmp_path / "probe.py"
    path.write_text(textwrap.dedent(script))
    clean = {k: v for k, v in os.environ.items() if not k.startswith("LCM_")}
    clean.update(env)
    return subprocess.run(
        [sys.executable, str(path)], env=clean, capture_output=True, text=True, timeout=180
    )


_PRELUDE = r"""
import importlib.util, os, sys, types
from pathlib import Path

PLUGIN = Path(os.environ["PLUGIN_DIR"])
ns = types.ModuleType("hermes_plugins"); ns.__path__ = []; sys.modules["hermes_plugins"] = ns

def load(name):
    spec = importlib.util.spec_from_file_location(
        f"hermes_plugins.{name}", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)])
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); return mod

class Ctx:
    engine = None
    def register_context_engine(self, e): self.engine = e
    def __getattr__(self, _n): return lambda *a, **k: None

def locks(db):
    pid = str(os.getpid()); ino = str(os.stat(db).st_ino)
    return sum(1 for l in open("/proc/locks") if l.split()[4] == pid and l.split()[5].endswith(":" + ino))

def register_in(mod, home):
    # Emulate the host binding a profile's home while it imports that profile's plugins.
    import hermes_constants_stub
    hermes_constants_stub.HOME = str(home)
    ctx = Ctx(); mod.register(ctx); return ctx.engine
"""

_STUB = r"""
import os
HOME = os.environ.get("HERMES_HOME", "")
def get_hermes_home():
    from pathlib import Path
    return Path(HOME)
def get_process_hermes_home():
    from pathlib import Path
    return Path(os.environ["HERMES_HOME"])
"""


def _stub_hermes(tmp_path: Path) -> Path:
    stub_dir = tmp_path / "stub"
    (stub_dir / "hermes_cli").mkdir(parents=True)
    (stub_dir / "hermes_constants_stub.py").write_text(_STUB)
    (stub_dir / "hermes_constants.py").write_text(
        "from hermes_constants_stub import get_hermes_home, get_process_hermes_home\n"
    )
    (stub_dir / "hermes_cli" / "__init__.py").write_text("")
    (stub_dir / "hermes_cli" / "config.py").write_text(
        "from hermes_constants_stub import get_hermes_home\n"
    )
    return stub_dir


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_foreign_profile_copy_uses_its_own_database_and_keeps_locks(tmp_path):
    root = tmp_path / "hermes"
    launch = root / "profiles" / "launch"
    other = root / "profiles" / "other"
    for home in (launch, other):
        home.mkdir(parents=True)
    # The launch profile's .env is ALSO in os.environ (systemd EnvironmentFile).
    (launch / ".env").write_text(f'LCM_DATABASE_PATH="{launch / "lcm.db"}"\nLCM_FRESH_TAIL_COUNT=7\n')
    (other / ".env").write_text("LCM_FRESH_TAIL_COUNT=11\n")
    stub = _stub_hermes(tmp_path)
    script = _PRELUDE + r"""
launch, other = Path(sys.argv[0]).parent / "hermes/profiles/launch", Path(sys.argv[0]).parent / "hermes/profiles/other"
first = load("hermes_lcm")
e1 = register_in(first, launch)
db1 = Path(e1._store.db_path)
assert db1 == launch / "lcm.db", db1
assert e1._config.fresh_tail_count == 7, e1._config.fresh_tail_count
before = locks(db1)
second = load("hermes_lcm__home_other")
e2 = register_in(second, other)
db2 = Path(e2._store.db_path)
print("db2", db2, "tail", e2._config.fresh_tail_count, "locks", before, locks(db1))
assert db2 == other / "lcm.db", db2
assert e2._config.fresh_tail_count == 11, e2._config.fresh_tail_count
assert before >= 1 and locks(db1) == before, (before, locks(db1))
print("OK")
"""
    env = {
        "PLUGIN_DIR": str(PLUGIN_DIR),
        "HERMES_HOME": str(launch),
        "LCM_DATABASE_PATH": str(launch / "lcm.db"),
        "LCM_FRESH_TAIL_COUNT": "7",
        "PYTHONPATH": os.pathsep.join([str(stub), os.environ.get("PYTHONPATH", "")]),
    }
    result = _run(script, env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_second_profile_copy_refused_on_same_database(tmp_path):
    root = tmp_path / "hermes"
    launch = root / "profiles" / "launch"
    other = root / "profiles" / "other"
    shared = tmp_path / "shared" / "lcm.db"
    shared.parent.mkdir()
    for home in (launch, other):
        home.mkdir(parents=True)
        (home / ".env").write_text(f'LCM_DATABASE_PATH="{shared}"\n')
    stub = _stub_hermes(tmp_path)
    script = _PRELUDE + r"""
base = Path(sys.argv[0]).parent
launch, other = base / "hermes/profiles/launch", base / "hermes/profiles/other"
first = load("hermes_lcm")
e1 = register_in(first, launch)
db = Path(e1._store.db_path)
before = locks(db)
second = load("hermes_lcm__home_other")
try:
    register_in(second, other)
except Exception as exc:
    print("refused:", type(exc).__name__)
else:
    raise SystemExit("second copy was allowed to open the same database")
assert locks(db) == before >= 1, (before, locks(db))
c = e1.clone_for_agent()   # same-profile clone stays allowed
assert locks(db) >= 1
print("OK")
"""
    env = {
        "PLUGIN_DIR": str(PLUGIN_DIR),
        "HERMES_HOME": str(launch),
        "PYTHONPATH": os.pathsep.join([str(stub), os.environ.get("PYTHONPATH", "")]),
    }
    result = _run(script, env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "refused: CrossProfileDatabaseError" in result.stdout
    assert "OK" in result.stdout


def test_foreign_profile_database_path_is_rebased(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "lcm_enforce_probe", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec is not None
    from types import SimpleNamespace

    root = tmp_path / "hermes"
    home = root / "profiles" / "b"
    module_globals: dict = {}
    src = (PLUGIN_DIR / "__init__.py").read_text()
    start = src.index("def _enforce_profile_owned_database")
    end = src.index("\ndef register(ctx)")
    exec("from pathlib import Path\n" + src[start:end], module_globals)
    enforce = module_globals["_enforce_profile_owned_database"]
    cases = {
        str(root / "profiles" / "a" / "lcm.db"): str(home / "lcm.db"),
        str(root / "lcm.db"): str(home / "lcm.db"),
        str(home / "lcm.db"): str(home / "lcm.db"),
        "/opt/data/lcm.db": "/opt/data/lcm.db",
    }
    for given, expected in cases.items():
        cfg = SimpleNamespace(database_path=given)
        enforce(cfg, str(home))
        assert cfg.database_path == expected, (given, cfg.database_path)
