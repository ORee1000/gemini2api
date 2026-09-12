"""启动时必须把「PSIDTS 轮换循环有没有启动」写进日志（issue #11，纯可观测性）。

_ensure_refresh_task() 只在 initialize() 拿不到 token 的 else 分支、以及 reload_cookies()
里被调用 —— 也就是说「正常启动」的网关从来不跑 _auto_refresh_loop。而这件事过去在日志里
一点痕迹都没有（没启动就什么都不打），下一次排查 PSIDTS 过期只能靠读源码推断。

这里只断言日志，并顺带钉死"只加日志、不改控制流"：两个分支各自该不该调
_ensure_refresh_task，以及 _healthy 的结果，都必须和加日志之前一模一样。
"""

import asyncio
import logging

import pytest

from app.core import gemini_client as gc


class _DummySession:
    def __init__(self, *a, **kw):
        pass


def _client(monkeypatch, tmp_path, *, token: str) -> tuple[gc.GeminiWebClient, list]:
    """造一个把所有外部 I/O 都摘掉的 client，只留 initialize() 的分支逻辑。"""
    monkeypatch.setattr(gc, "AsyncSession", _DummySession)
    monkeypatch.setattr(gc.fingerprint_config, "load", lambda: None)
    monkeypatch.setattr(gc.header_builder, "get_impersonate_target", lambda: "chrome120")
    monkeypatch.setattr(gc.settings, "health_check_enabled", False)
    monkeypatch.setattr(gc, "PersistentCookieJar", lambda psid: _DummyJar())

    client = gc.GeminiWebClient(psid="PSID", psidts="PSIDTS")

    async def _obtain():
        client._session_token = token

    async def _heartbeat():
        return None

    ensured = []
    monkeypatch.setattr(client, "_obtain_session_token", _obtain)
    monkeypatch.setattr(client, "_send_heartbeat", _heartbeat)

    # 只拦住"真的创建后台任务"这一步，_ensure_refresh_task 本体照跑 —— 它自己也会打一句
    # "Auto-refresh loop started"。以前这里把整个方法 stub 掉，于是测试看不见它的日志，
    # 生产上两句一起打（重复条目）也照样全绿。
    def _fake_create_task(coro, *a, **kw):
        coro.close()  # 别让协程留下 "never awaited" 警告
        return _DummyTask()

    monkeypatch.setattr(gc.asyncio, "create_task", _fake_create_task)

    real_ensure = client._ensure_refresh_task

    def _spy():
        ensured.append(True)
        return real_ensure()

    monkeypatch.setattr(client, "_ensure_refresh_task", _spy)
    return client, ensured


class _DummyTask:
    def done(self):
        return False

    def cancel(self):
        return None


class _DummyJar:
    def get_all(self):
        return {}

    def set(self, *a, **kw):
        return None


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.fixture
def _caplog_info(caplog):
    caplog.set_level(logging.INFO, logger=gc.logger.name)
    return caplog


def test_healthy_startup_says_the_loop_is_not_running(monkeypatch, tmp_path, _caplog_info):
    client, ensured = _client(monkeypatch, tmp_path, token="SNlM0e-value")
    asyncio.run(client.initialize())

    msgs = _messages(_caplog_info)
    hits = [m for m in msgs if "Auto-refresh loop" in m]
    assert len(hits) == 1, f"启动必须恰好说一次轮换循环的状态，实际: {hits}"
    assert "NOT started" in hits[0]
    assert "startup token OK" in hits[0], "还得说清楚为什么没启动"

    # 控制流未变：拿到 token 时本来就不该启动轮换循环
    assert ensured == []
    assert client.is_healthy is True


def test_tokenless_startup_says_the_loop_started_and_why(monkeypatch, tmp_path, _caplog_info):
    client, ensured = _client(monkeypatch, tmp_path, token="")
    asyncio.run(client.initialize())

    msgs = _messages(_caplog_info)
    hits = [m for m in msgs if "Auto-refresh loop" in m]
    assert len(hits) == 1, f"启动必须恰好说一次轮换循环的状态（不许重复），实际: {hits}"
    assert "NOT started" not in hits[0]
    assert "started" in hits[0]
    # 原因由紧挨着的上一条 warning 说清楚，不再重复打一句 started
    assert any("Token not found" in m for m in msgs), "还得说清楚为什么启动了"

    # 控制流未变：拿不到 token 才调 _ensure_refresh_task
    assert ensured == [True]
    assert client.is_healthy is False


def test_the_two_branches_do_not_log_the_same_thing(monkeypatch, tmp_path, _caplog_info):
    """两条日志必须互相可分辨，否则等于没加。"""
    ok, _ = _client(monkeypatch, tmp_path, token="SNlM0e-value")
    asyncio.run(ok.initialize())
    healthy_line = [m for m in _messages(_caplog_info) if "Auto-refresh loop" in m][0]

    _caplog_info.clear()
    bad, _ = _client(monkeypatch, tmp_path, token="")
    asyncio.run(bad.initialize())
    tokenless_line = [m for m in _messages(_caplog_info) if "Auto-refresh loop" in m][0]

    assert healthy_line != tokenless_line
