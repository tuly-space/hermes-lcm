"""Background workers keep the caller's context (the Hermes profile).

In a multiplexed gateway the active profile lives in context variables
(HERMES_HOME override, secret scope). A plain ``threading.Thread`` starts
with an empty context, so an unbound worker resolves the gateway's default
profile: wrong auxiliary provider, wrong secrets, wrong auth.
"""

from __future__ import annotations

import contextvars
import threading

import hermes_lcm.engine as engine_mod
import hermes_lcm.tools as lcm_tools

PROFILE = contextvars.ContextVar("lcm_test_profile", default="default")


def test_plain_thread_loses_context_baseline():
    token = PROFILE.set("jamie")
    try:
        seen = []
        worker = threading.Thread(target=lambda: seen.append(PROFILE.get()))
        worker.start()
        worker.join(5)
        assert seen == ["default"]
    finally:
        PROFILE.reset(token)


def test_rollup_scheduler_runs_job_in_scheduling_context():
    scheduler = engine_mod._RollupMaintenanceScheduler()
    seen = []
    done = threading.Event()

    def job() -> None:
        seen.append(PROFILE.get())
        done.set()

    token = PROFILE.set("jamie")
    try:
        assert scheduler.schedule(("db", "scope"), job)
    finally:
        PROFILE.reset(token)
    assert done.wait(5)
    assert seen == ["jamie"]


def test_rollup_jobs_from_two_profiles_keep_their_own_context():
    scheduler = engine_mod._RollupMaintenanceScheduler()
    seen = {}
    finished = {"jamie": threading.Event(), "lanlan": threading.Event()}

    def job_for(name: str):
        def job() -> None:
            seen[name] = PROFILE.get()
            finished[name].set()
        return job

    for name in ("jamie", "lanlan"):
        token = PROFILE.set(name)
        try:
            assert scheduler.schedule((name, "scope"), job_for(name))
        finally:
            PROFILE.reset(token)
    assert all(event.wait(5) for event in finished.values())
    assert seen == {"jamie": "jamie", "lanlan": "lanlan"}


def test_deadline_worker_runs_in_caller_context():
    token = PROFILE.set("hermes-app-dev")
    try:
        result = lcm_tools._run_within_deadline(
            PROFILE.get, remaining_s=5, name="lcm-test-worker",
            worker_slots=threading.BoundedSemaphore(1),
        )
    finally:
        PROFILE.reset(token)
    assert result == "hermes-app-dev"
