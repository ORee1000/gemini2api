"""面板必须暴露有鉴别力的账号健康信息（issue #11）。

issue #11 的截图上账号显示 ACTIVE / Errors 0 / Models 4 / Requests 4，聊天框却一个字
都没有 —— 这四个数字没有一个有鉴别力：
- status 只有连挂 3 次或撞 401/403 才变 EXPIRED，cookie 死透的账号照样绿着；
- models_count 只要 client 对象在就恒等于公开模型数；
- request_count 失败也 +1。
所以 get_status() 必须把 is_healthy / consecutive_failures / last_success_at 导出来，
last_success_at 还必须是「真的成功过」的时间，不能拿 last_used（派活时间）冒充。
"""

import asyncio
import json
import re
import shutil
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.account_pool import (
    Account,
    AccountPool,
    AccountStatus,
    _error_summary,
)
from app.core.gemini_client import HTTPStatusError

_ROOT = Path(__file__).resolve().parents[2]
_APP_JS = _ROOT / "static" / "app" / "app.js"
_I18N_JS = _ROOT / "static" / "app" / "i18n.js"
_LANGS = ("zh-CN", "en-US", "ja-JP", "ko-KR", "zh-TW")

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node 不可用，跳过前端行为测试")


class _FakeClient:
    def __init__(self, healthy: bool = True):
        self._healthy = healthy

    @property
    def is_healthy(self) -> bool:
        return self._healthy


def _pool(*accounts: Account) -> AccountPool:
    pool = AccountPool.__new__(AccountPool)
    pool._accounts = list(accounts)
    pool._cond = asyncio.Condition()
    pool._strategy = type("S", (), {"value": "round-robin"})()
    pool._max_concurrent = 3
    return pool


def _acc(**kw) -> Account:
    base = dict(id="account-0", psid="p", psidts="t", label="L")
    base.update(kw)
    return Account(**base)


# --------------------------------------------------------------------------
# get_status 导出
# --------------------------------------------------------------------------

def test_get_status_exposes_the_fields_that_actually_discriminate():
    a = _acc(client=_FakeClient(healthy=False), consecutive_failures=2)
    a.last_success_at = datetime(2026, 9, 12, 3, 4, 5, tzinfo=timezone.utc)
    info = _pool(a).get_status()["accounts"][0]

    assert info["is_healthy"] is False
    assert info["consecutive_failures"] == 2
    assert info["last_success_at"] == "2026-09-12T03:04:05+00:00"


def test_healthy_client_reports_is_healthy_true():
    info = _pool(_acc(client=_FakeClient(healthy=True))).get_status()["accounts"][0]
    assert info["is_healthy"] is True


def test_account_without_client_is_not_healthy():
    """client 还没建起来（初始化失败）的账号不能报 healthy。"""
    info = _pool(_acc(client=None)).get_status()["accounts"][0]
    assert info["is_healthy"] is False


def test_never_used_account_reports_null_last_success():
    info = _pool(_acc(client=_FakeClient())).get_status()["accounts"][0]
    assert info["last_success_at"] is None


def test_last_success_is_not_last_used():
    """核心鉴别力：派过活但从没成功过的账号，last_used 有值、last_success_at 必须仍为 None。

    这正是 issue #11 的形态：ACTIVE + Requests 一直涨 + 从来没成功过。
    """
    a = _acc(client=_FakeClient(), request_count=1247, status=AccountStatus.ACTIVE)
    a.last_used = datetime.now(timezone.utc)
    info = _pool(a).get_status()["accounts"][0]

    assert info["last_used"] is not None
    assert info["last_success_at"] is None


def test_status_export_keeps_existing_keys():
    """零回归：既有键一个都不能少（面板现有渲染依赖它们）。"""
    info = _pool(_acc(client=_FakeClient())).get_status()["accounts"][0]
    for key in ("id", "label", "psid", "status", "request_count", "error_count",
                "active_requests", "last_used", "cooling_down", "models", "models_count"):
        assert key in info, f"既有导出键 {key} 丢了"


def test_get_status_works_with_empty_pool():
    assert _pool().get_status()["accounts"] == []


def test_get_status_survives_being_called_outside_an_event_loop():
    """纯只读的状态导出不该因为"当前线程没有运行中的 event loop"而崩掉。"""
    a = _acc(client=_FakeClient())
    pool = _pool(a)
    asyncio.run(pool.release(a, success=True))  # 跑完后当前线程已无 event loop

    info = pool.get_status()["accounts"][0]
    assert info["cooling_down"] is False
    assert info["last_success_at"] is not None


# --------------------------------------------------------------------------
# release 对 last_success_at / last_error 的写入
# --------------------------------------------------------------------------

def test_release_success_stamps_last_success_at():
    a = _acc(client=_FakeClient(), active_requests=1)
    pool = _pool(a)
    before = datetime.now(timezone.utc)
    asyncio.run(pool.release(a, success=True))

    assert a.last_success_at is not None
    assert before - timedelta(seconds=5) <= a.last_success_at <= datetime.now(timezone.utc) + timedelta(seconds=5)


def test_release_failure_does_not_stamp_last_success_at():
    a = _acc(client=_FakeClient(), active_requests=1)
    asyncio.run(_pool(a).release(a, success=False, error="X"))
    assert a.last_success_at is None


def test_release_failure_records_redacted_summary():
    a = _acc(client=_FakeClient(), active_requests=1)
    pool = _pool(a)
    asyncio.run(pool.release(a, success=False, error=_error_summary(HTTPStatusError(503, "boom"))))

    assert a.last_error == "HTTPStatusError (HTTP 503)"
    assert a.last_error_at is not None
    assert pool.get_status()["accounts"][0]["last_error"] == "HTTPStatusError (HTTP 503)"


def test_release_without_error_argument_leaves_last_error_untouched():
    """零回归：既有调用点不传 error，不得把已有的 last_error 抹掉。"""
    a = _acc(client=_FakeClient(), active_requests=1, last_error="HTTPStatusError (HTTP 401)")
    asyncio.run(_pool(a).release(a, success=False))
    assert a.last_error == "HTTPStatusError (HTTP 401)"


# --------------------------------------------------------------------------
# 脱敏
# --------------------------------------------------------------------------

def test_error_summary_drops_upstream_body():
    """HTTPStatusError 的消息里嵌着 Google 响应正文前 200 字节，绝不能落进 last_error。"""
    leak = 'HTTP 500: <html>"SNlM0e":"AJz9-secret-token" victim@example.com</html>'
    summary = _error_summary(HTTPStatusError(500, leak))

    assert summary == "HTTPStatusError (HTTP 500)"
    assert "SNlM0e" not in summary
    assert "victim@example.com" not in summary


def test_error_summary_drops_client_controlled_model_name():
    """ValueError 的消息里嵌着请求里的 model 名（客户端可控），是面板存储型 XSS 的入口。"""
    summary = _error_summary(ValueError("Model '<img src=x onerror=alert(1)>' unavailable"))

    assert summary == "ValueError"
    assert "<img" not in summary


def test_error_summary_still_discriminates_between_failure_kinds():
    """脱敏不能脱到没信息：三类故障必须还能分辨。"""
    kinds = {
        _error_summary(HTTPStatusError(401, "x")),
        _error_summary(HTTPStatusError(503, "x")),
        _error_summary(RuntimeError("Client not ready")),
    }
    assert len(kinds) == 3, kinds


# --------------------------------------------------------------------------
# 前端：健康提示（跑 app.js 里真正的那个纯函数）
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
    fn = _extract_function(src, "_accountHealthNote")
    assert fn.strip().startswith("function _accountHealthNote("), "没切到 _accountHealthNote"
    script = tmp_path / "note.mjs"
    script.write_text(
        fn + textwrap.dedent("""
        function t(k) { return k; }
        process.stdout.write(JSON.stringify(_accountHealthNote(JSON.parse(process.argv[2]))));
        """),
        encoding="utf-8",
    )
    proc = subprocess.run([_NODE, str(script), json.dumps(account)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@_needs_node
def test_active_but_unhealthy_account_gets_a_loud_note(tmp_path):
    """面板骗人的那一格：status=active 但 is_healthy=false，必须醒目标出来。"""
    note = _note({"status": "active", "is_healthy": False, "consecutive_failures": 0}, tmp_path)

    assert note is not None
    assert note["cls"] == "text-danger"
    assert note["text"] == "accounts.healthMismatch"


@_needs_node
def test_expired_and_unhealthy_is_not_screamed_about(tmp_path):
    """状态已经是 expired 时两边一致，不需要醒目红字。"""
    note = _note({"status": "expired", "is_healthy": False, "consecutive_failures": 0}, tmp_path)

    assert note["cls"] == "text-muted"
    assert note["text"] == "accounts.sessionUnhealthy"


@_needs_node
def test_consecutive_failures_are_surfaced(tmp_path):
    note = _note({"status": "active", "is_healthy": True, "consecutive_failures": 2}, tmp_path)

    assert note["cls"] == "text-warning"
    assert note["text"] == "accounts.consecutiveFailures: 2"


@_needs_node
def test_healthy_account_gets_no_note(tmp_path):
    """健康且无连续失败时不加噪音。"""
    assert _note({"status": "active", "is_healthy": True, "consecutive_failures": 0}, tmp_path) is None


@_needs_node
def test_missing_health_field_is_not_treated_as_unhealthy(tmp_path):
    """老版本后端（还没这个字段）不该被误报成不健康。"""
    assert _note({"status": "active"}, tmp_path) is None


# --------------------------------------------------------------------------
# i18n 齐备
# --------------------------------------------------------------------------

def _i18n_account_keys() -> dict:
    blocks = {}
    current = None
    for line in _I18N_JS.read_text(encoding="utf-8").splitlines():
        lang = re.match(r"^ {4}'([A-Za-z-]+)': \{$", line)
        if lang:
            current = lang.group(1)
            blocks[current] = set()
            continue
        if current is None:
            continue
        key = re.match(r"^ +'(accounts\.[^']+)':", line)
        if key:
            blocks[current].add(key.group(1))
    assert set(_LANGS) <= set(blocks), f"i18n.js 语言块解析异常: {sorted(blocks)}"
    return blocks


def test_new_health_keys_exist_in_every_language():
    needed = {
        "accounts.health", "accounts.lastSuccess", "accounts.lastError",
        "accounts.healthMismatch", "accounts.sessionUnhealthy", "accounts.consecutiveFailures",
    }
    blocks = _i18n_account_keys()
    missing = {lang: sorted(needed - blocks[lang]) for lang in _LANGS if needed - blocks[lang]}
    assert not missing, f"这些语言块缺少账号健康相关的 i18n 键: {missing}"


def test_all_language_blocks_share_the_same_account_keys():
    blocks = _i18n_account_keys()
    baseline = blocks["zh-CN"]
    assert baseline, "没解析到任何 accounts.* 键，i18n.js 格式可能变了"
    drift = {
        lang: {"missing": sorted(baseline - blocks[lang]), "extra": sorted(blocks[lang] - baseline)}
        for lang in _LANGS
        if blocks[lang] != baseline
    }
    assert not drift, f"accounts.* 键在各语言块之间漂移了: {drift}"
