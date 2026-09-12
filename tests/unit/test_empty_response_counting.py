"""空响应必须可计数（issue #11）。

以前：上游 HTTP 200 但正文解析不出任何内容时，generate/generate_stream 静默产出空结果、
不抛异常，于是走 release(success=True) —— Errors 纹丝不动，consecutive_failures 还被清零。
结果是"每次都吐空"的账号在面板上完全看不出来，正是 issue #11 截图里 Errors 0 的由来。

现在：收尾后判一次空，单独计数；单次绝不降级账号，连续到阈值才在面板可见的 Errors 上留痕。
"""

import asyncio
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from app.core.account_pool import (
    EMPTY_STREAK_THRESHOLD,
    Account,
    AccountPool,
    AccountStatus,
    _is_empty_generation,
)

_APP_JS = Path(__file__).resolve().parents[2] / "static" / "app" / "app.js"
_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node 不可用，跳过前端行为测试")


def _acc(**kw) -> Account:
    base = dict(id="account-0", psid="p", psidts="t", label="L")
    base.update(kw)
    return Account(**base)


def _pool(*accounts: Account) -> AccountPool:
    pool = AccountPool.__new__(AccountPool)
    pool._accounts = list(accounts)
    pool._cond = asyncio.Condition()
    pool._strategy = type("S", (), {"value": "round-robin"})()
    pool._max_concurrent = 3
    return pool


async def _release(pool, account, **kw):
    await pool.release(account, success=True, **kw)


# --------------------------------------------------------------------------
# 判空口径
# --------------------------------------------------------------------------

def test_no_text_no_images_is_empty():
    assert _is_empty_generation({"text": "", "images": [], "conversation_id": "c"}) is True
    assert _is_empty_generation({"text": "   ", "images": []}) is True


def test_missing_final_event_is_empty():
    """流跑完了却从没发过 final 事件（上游中途静默截断）也是空。"""
    assert _is_empty_generation(None) is True


def test_text_only_is_not_empty():
    assert _is_empty_generation({"text": "hi", "images": []}) is False


def test_images_only_is_not_empty():
    """只有图片没有文字是正常的生图回复，绝不能算空。"""
    assert _is_empty_generation({"text": "", "images": [{"mime": "image/png", "b64": "x"}]}) is False


def test_thoughts_only_is_not_empty():
    """只有思考链没有正文：上游确实生成了东西、会话是活的，不记空（宁可漏报不误伤）。"""
    assert _is_empty_generation({"text": "", "images": [], "thoughts": "let me think"}) is False


def test_blank_thoughts_do_not_rescue_an_empty_result():
    assert _is_empty_generation({"text": "", "images": [], "thoughts": "   "}) is True


# --------------------------------------------------------------------------
# 计数与降级边界
# --------------------------------------------------------------------------

def test_one_and_two_empties_do_not_degrade_the_account():
    a = _acc(active_requests=2)
    pool = _pool(a)

    async def go():
        await _release(pool, a, empty=True)
        assert a.consecutive_empty == 1
        assert a.error_count == 0, "单次空响应绝不能计进 Errors"
        await _release(pool, a, empty=True)
        assert a.consecutive_empty == 2
        assert a.error_count == 0

    asyncio.run(go())
    assert a.status == AccountStatus.ACTIVE
    assert a.consecutive_failures == 0
    assert a.cooldown_until == 0.0
    assert a.empty_count == 2


def test_third_consecutive_empty_shows_up_in_errors():
    a = _acc(active_requests=3)
    pool = _pool(a)

    async def go():
        for _ in range(EMPTY_STREAK_THRESHOLD):
            await _release(pool, a, empty=True)

    asyncio.run(go())
    assert a.consecutive_empty == EMPTY_STREAK_THRESHOLD
    assert a.error_count == 1
    assert "Empty response x3" in a.last_error
    assert a.last_error_at is not None
    # 但依然不降级：空响应自己不足以判定账号已死
    assert a.status == AccountStatus.ACTIVE
    assert a.consecutive_failures == 0


def test_errors_keep_growing_while_stuck_on_empty():
    """卡在空响应里的账号 Errors 应持续增长，面板才看得出还在恶化。"""
    a = _acc(active_requests=5)
    pool = _pool(a)

    async def go():
        for _ in range(5):
            await _release(pool, a, empty=True)

    asyncio.run(go())
    assert a.error_count == 3, "第 3/4/5 次各 +1"
    assert a.empty_count == 5


def test_one_non_empty_success_resets_the_streak():
    a = _acc(active_requests=4)
    pool = _pool(a)

    async def go():
        await _release(pool, a, empty=True)
        await _release(pool, a, empty=True)
        await _release(pool, a, empty=False)   # 中间夹一次非空成功
        assert a.consecutive_empty == 0
        await _release(pool, a, empty=True)
        assert a.consecutive_empty == 1

    asyncio.run(go())
    assert a.error_count == 0, "被打断的 streak 不该攒到阈值"
    assert a.empty_count == 3, "总数仍然累计"


def test_empty_release_does_not_stamp_last_success():
    """核心：吐了个空不能算"会话还活着"的证据。"""
    a = _acc(active_requests=1)
    asyncio.run(_release(_pool(a), a, empty=True))
    assert a.last_success_at is None


def test_empty_release_does_not_clear_real_failures():
    """空响应不该替真实的连续失败擦屁股。"""
    a = _acc(active_requests=1, consecutive_failures=2)
    asyncio.run(_release(_pool(a), a, empty=True))
    assert a.consecutive_failures == 2


def test_non_empty_release_keeps_its_old_behaviour():
    """零回归：不传 empty 时与改动前逐字节等价。"""
    a = _acc(active_requests=1, consecutive_failures=2, consecutive_empty=2)
    pool = _pool(a)
    asyncio.run(pool.release(a, success=True))

    assert a.consecutive_failures == 0
    assert a.consecutive_empty == 0
    assert a.error_count == 0
    assert a.last_success_at is not None


def test_failure_path_never_touches_empty_counters():
    a = _acc(active_requests=2, consecutive_empty=1)
    pool = _pool(a)

    async def go():
        await pool.release(a, success=False)
        await pool.release_disconnected(a)

    asyncio.run(go())
    assert a.consecutive_empty == 1, "失败/断连都不是空响应，不得累加也不得清零"
    assert a.empty_count == 0


def test_cooldown_path_never_touches_empty_counters():
    """5xx 限流是另一回事，不得蹭空响应计数器。"""
    a = _acc(active_requests=1, consecutive_empty=1)
    asyncio.run(_pool(a).release(a, success=False, cooldown=True))

    assert a.consecutive_empty == 1
    assert a.empty_count == 0


def test_empty_counters_are_exported_to_the_panel():
    a = _acc(active_requests=1)
    pool = _pool(a)
    asyncio.run(_release(pool, a, empty=True))

    info = pool.get_status()["accounts"][0]
    assert info["consecutive_empty"] == 1
    assert info["empty_count"] == 1


# --------------------------------------------------------------------------
# 走完整的 generate / generate_stream（判空必须在收尾之后）
# --------------------------------------------------------------------------

class _ScriptedClient:
    """按脚本产出结果的假 client。"""

    def __init__(self, result=None, events=None):
        self._result = result
        self._events = events or []

    async def generate(self, *a, **kw):
        return self._result

    async def generate_stream(self, *a, **kw):
        for evt in self._events:
            yield evt


def _pinned_pool(client) -> tuple[AccountPool, Account]:
    a = _acc(client=client, active_requests=0)
    pool = _pool(a)

    async def _acquire(exclude=None):
        a.active_requests += 1
        return a

    pool.acquire = _acquire
    return pool, a


def test_generate_counts_an_empty_upstream_200():
    pool, a = _pinned_pool(_ScriptedClient(result={"text": "", "images": [], "conversation_id": "c"}))
    asyncio.run(pool.generate("hi", "gemini-3-flash", account_id="account-0"))

    assert a.consecutive_empty == 1
    assert a.last_success_at is None


def test_generate_does_not_count_a_real_answer():
    pool, a = _pinned_pool(_ScriptedClient(result={"text": "hello", "images": []}))
    asyncio.run(pool.generate("hi", "gemini-3-flash", account_id="account-0"))

    assert a.consecutive_empty == 0
    assert a.empty_count == 0
    assert a.last_success_at is not None


def test_stream_judges_only_after_the_final_event():
    """中途每一帧都可能没文本 —— 在流中途判空会把正常回复误记成空。"""
    events = [
        {"type": "delta", "text": ""},           # 中途空帧，不能据此判空
        {"type": "delta", "text": "answer"},
        {"type": "final", "text": "answer", "images": [], "conversation_id": "c"},
    ]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 0
    assert a.empty_count == 0
    assert a.last_success_at is not None


def test_stream_with_only_an_empty_final_is_counted():
    events = [{"type": "final", "text": "", "images": [], "conversation_id": "c"}]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 1
    assert a.empty_count == 1


def test_stream_with_thoughts_only_is_not_counted_as_empty():
    events = [
        {"type": "thoughts", "text": "hmm"},
        {"type": "final", "text": "", "images": [], "thoughts": "hmm", "conversation_id": "c"},
    ]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 0
    assert a.empty_count == 0


def test_stream_that_streamed_text_but_lost_its_final_is_not_counted():
    """内容确实吐给用户了，只是 final 事件没来（断尾）—— 用户已经看到回复，不算空。"""
    events = [{"type": "delta", "text": "answer"}]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 0
    assert a.empty_count == 0
    assert a.last_success_at is not None


def test_stream_that_only_streamed_thoughts_without_final_is_not_counted():
    """思考链已经流出去了，即使 final 没带上 thoughts 也不算空。"""
    events = [{"type": "thoughts", "text": "hmm"},
              {"type": "final", "text": "", "images": [], "conversation_id": "c"}]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 0


def test_stream_that_never_sends_final_is_counted():
    """上游中途静默截断：一个 final 都没来。"""
    events = [{"type": "delta", "text": ""}]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 1


def test_stream_with_images_only_is_not_counted_as_empty():
    events = [{"type": "final", "text": "", "images": [{"mime": "image/png", "b64": "x"}],
               "conversation_id": "c"}]
    pool, a = _pinned_pool(_ScriptedClient(events=events))
    asyncio.run(_drain(pool))

    assert a.consecutive_empty == 0
    assert a.last_success_at is not None


async def _drain(pool):
    out = []
    async for evt in pool.generate_stream("hi", "gemini-3-flash", account_id="account-0"):
        out.append(evt)
    return out


# --------------------------------------------------------------------------
# 前端空响应提示（跑 app.js 里真正的那个纯函数）
# --------------------------------------------------------------------------

def _extract_function(source: str, name: str) -> str:
    head = source.index(f"function {name}(")
    depth = 0
    i = source.index("{", head)
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[head:i + 1]
        i += 1
    raise AssertionError(f"{name} 的大括号没配对，app.js 格式可能变了")


def _note(account: dict, tmp_path: Path):
    src = _APP_JS.read_text(encoding="utf-8")
    fn = _extract_function(src, "_accountEmptyNote")
    assert fn.strip().startswith("function _accountEmptyNote("), "没切到 _accountEmptyNote"
    assert "const EMPTY_STREAK_THRESHOLD = 3;" in src, "前端阈值常量没了，app.js 格式可能变了"
    script = tmp_path / "empty.mjs"
    script.write_text(
        "const EMPTY_STREAK_THRESHOLD = 3;\n" + fn + textwrap.dedent("""
        function t(k) { return k; }
        process.stdout.write(JSON.stringify(_accountEmptyNote(JSON.parse(process.argv[2]))));
        """),
        encoding="utf-8",
    )
    proc = subprocess.run([_NODE, str(script), json.dumps(account)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@_needs_node
def test_account_that_never_returned_empty_gets_no_row(tmp_path):
    assert _note({"empty_count": 0, "consecutive_empty": 0}, tmp_path) is None
    assert _note({}, tmp_path) is None


@_needs_node
def test_empty_row_shows_total_and_current_streak(tmp_path):
    note = _note({"empty_count": 7, "consecutive_empty": 2}, tmp_path)
    assert note["text"] == "7 (accounts.consecutiveNow 2)"
    assert note["cls"] == "text-warning"


@_needs_node
def test_streak_at_threshold_turns_the_row_red(tmp_path):
    note = _note({"empty_count": 9, "consecutive_empty": 3}, tmp_path)
    assert note["cls"] == "text-danger"


@_needs_node
def test_recovered_account_shows_history_without_alarm(tmp_path):
    """空过但已经恢复（streak 归零）：显示历史总数，不报警。"""
    note = _note({"empty_count": 4, "consecutive_empty": 0}, tmp_path)
    assert note["text"] == "4"
    assert note["cls"] == "text-warning"
