"""Threshold-only summary regressions: no real provider/network calls."""
import json
import socket
from unittest.mock import Mock

import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('network forbidden')))
    engines = []
    def build(**kw):
        cfg = dict(database_path=str(tmp_path / f'{len(engines)}.db'),
                   context_threshold=.8, fresh_tail_count=4, leaf_chunk_tokens=100,
                   automatic_summary_threshold_only=True)
        cfg.update(kw)
        engine = LCMEngine(config=LCMConfig(**cfg), hermes_home=str(tmp_path))
        engine.on_session_start('test-session', context_length=272000)
        engines.append(engine)
        return engine
    yield build
    for engine in engines:
        engine.shutdown()


def backlog():
    return [{'role': 'user' if i % 2 == 0 else 'assistant',
             'content': f'Unique evidence {i}: ' + ('context detail ' * 100)} for i in range(30)]


def spy_summary(monkeypatch):
    spy = Mock(side_effect=RuntimeError('summary entered'))
    # Other suites reload hermes_lcm.engine; patch the globals this class uses.
    monkeypatch.setitem(LCMEngine.__init__.__globals__, 'summarize_with_escalation', spy)
    return spy


@pytest.mark.parametrize('current', [None, 1000, 217599])
def test_below_threshold_direct_and_stale_usage(make_engine, monkeypatch, current):
    e = make_engine(); ms = backlog(); spy = spy_summary(monkeypatch)
    e.last_prompt_tokens = 260000
    out = e.compress(ms, current_tokens=current)
    assert out is ms
    assert e.last_compression_status == 'noop'
    spy.assert_not_called()
    assert e._store.get_session_count(e._session_id) == len(ms)


@pytest.mark.parametrize('kind', ['manual', 'threshold', 'overflow', 'manual_after_cleanup_latch'])
def test_explicit_threshold_and_overflow_preserved(make_engine, monkeypatch, kind):
    e = make_engine(max_assembly_tokens=10000 if kind == 'overflow' else 0)
    ms = backlog(); spy = spy_summary(monkeypatch)
    if kind == 'manual_after_cleanup_latch':
        e._preflight_cleanup_only_due_to_boundary_cooldown = True
    kwargs = {'force': True} if kind.startswith('manual') else {'current_tokens': 217600 if kind == 'threshold' else 10000}
    with pytest.raises(RuntimeError, match='summary entered'):
        e.compress(ms, **kwargs)
    assert spy.call_count == 1


def test_metadata_only_preflight_repeated(make_engine, monkeypatch):
    e = make_engine(); ms = backlog(); e.ingest(ms)
    spy = spy_summary(monkeypatch)
    for i in range(5):
        restored = [dict(m, _row_id=j+1, timestamp=10000+i+j, _db_persisted=True) for j, m in enumerate(ms)]
        assert not e.should_compress_preflight(restored)
        out = e.compress(restored, current_tokens=50000)
        assert count_messages_tokens(out) == count_messages_tokens(restored)
    spy.assert_not_called()


def test_large_tool_cleanup_preserves_tool_pair_without_summary(make_engine, monkeypatch):
    e = make_engine(large_output_externalization_enabled=True,
                    large_output_active_replay_stubbing_enabled=True,
                    large_output_active_replay_stub_threshold_tokens=100)
    ms = backlog() + [
        {'role':'assistant','content':'Read archive','tool_calls':[{'id':'call-a','type':'function','function':{'name':'read_file','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'call-a','content':'text result ' * 5000}]
    spy = spy_summary(monkeypatch)
    assert e.should_compress_preflight(ms)
    out = e.compress(ms, current_tokens=50000)
    assert e.last_compression_status == 'sanitized'
    assert any(m.get('tool_call_id') == 'call-a' for m in out)
    assert any(tc['id'] == 'call-a' for m in out for tc in m.get('tool_calls', []))
    spy.assert_not_called()


def test_switches_default_off_and_environment(monkeypatch):
    assert LCMConfig().automatic_summary_threshold_only is False
    monkeypatch.setenv('LCM_AUTOMATIC_SUMMARY_THRESHOLD_ONLY', 'true')
    cfg = LCMConfig.from_env()
    assert cfg.automatic_summary_threshold_only


def test_host_maintenance_passes_full_request_pressure(make_engine, monkeypatch):
    from types import SimpleNamespace
    from agent.turn_context_compaction import CompactionOutcome, _engine_preflight_maintenance
    e = make_engine(large_output_externalization_enabled=True,
                    large_output_active_replay_stubbing_enabled=True,
                    large_output_active_replay_stub_threshold_tokens=100)
    ms = backlog() + [
        {'role':'assistant','content':'Read archive','tool_calls':[{'id':'call-a','type':'function','function':{'name':'read_file','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'call-a','content':'text result ' * 5000}]
    spy = spy_summary(monkeypatch)
    def host_compress(messages, system, approx_tokens, task_id):
        assert approx_tokens == 217600
        return e.compress(messages, current_tokens=approx_tokens), system
    host = SimpleNamespace(_compress_context=host_compress)
    out = CompactionOutcome(messages=ms, active_system_prompt='', conversation_history=ms, current_turn_user_idx=0)
    with pytest.raises(RuntimeError, match='summary entered'):
        _engine_preflight_maintenance(host, out, e, 217600, '', 'test')
    assert spy.call_count == 1


def test_host_metadata_only_no_maintenance(make_engine, monkeypatch):
    from types import SimpleNamespace
    from agent.turn_context_compaction import CompactionOutcome, _engine_preflight_maintenance
    e = make_engine(); ms = backlog(); e.ingest(ms)
    restored = [dict(m, timestamp=1000+i, _db_persisted=True) for i,m in enumerate(ms)]
    host = SimpleNamespace(_compress_context=Mock(side_effect=AssertionError('unnecessary host compress')))
    out = CompactionOutcome(messages=restored, active_system_prompt='', conversation_history=restored, current_turn_user_idx=0)
    _engine_preflight_maintenance(host, out, e, 50000, '', 'test')
    host._compress_context.assert_not_called()
    assert out.messages is restored and not out.compressed
