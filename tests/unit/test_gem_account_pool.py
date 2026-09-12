import asyncio
import types
import pytest
from unittest.mock import AsyncMock, patch
from app.core.account_pool import AccountPool


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def list_gems(self):
        return [{"id": "g1", "name": "n", "description": "", "prompt": ""}]

    async def create_gem(self, name, prompt, description=""):
        return "new-id"

    async def update_gem(self, gem_id, name, prompt, description=""):
        self.calls.append(("update_gem", gem_id, name, prompt, description))
        return True

    async def delete_gem(self, gem_id):
        self.calls.append(("delete_gem", gem_id))
        return True

    async def generate(self, prompt, model, conversation_id="", attachments=None, gem_id=None,
                       extended_thinking=False):
        self.calls.append(("generate", prompt, model, gem_id))
        return {"text": "ok", "images": [], "conversation_id": "c1"}

    async def generate_stream(self, prompt, model, conversation_id="", attachments=None, gem_id=None,
                              extended_thinking=False):
        self.calls.append(("generate_stream", prompt, model, gem_id))
        yield {"type": "delta", "text": "hello"}
        yield {"type": "final", "text": "hello", "images": [], "conversation_id": "c2"}


def _pool_with_accounts():
    pool = AccountPool.__new__(AccountPool)
    a0 = types.SimpleNamespace(id="account-0", client=_FakeClient())
    a1 = types.SimpleNamespace(id="account-1", client=_FakeClient())
    pool._accounts = [a0, a1]
    return pool, a0, a1


def test_get_account_by_id():
    pool, a0, a1 = _pool_with_accounts()
    assert pool._get_account("account-1") is a1
    assert pool._get_account("nope") is None


def test_list_gems_routes_to_account():
    pool, a0, a1 = _pool_with_accounts()
    gems = asyncio.run(pool.list_gems("account-0"))
    assert gems[0]["id"] == "g1"


def test_list_gems_unknown_account_raises():
    pool, a0, a1 = _pool_with_accounts()
    with pytest.raises(ValueError):
        asyncio.run(pool.list_gems("ghost"))


def test_generate_pinned_account_passes_gem_id():
    """验证锁定逻辑：account_id 固定时，acquire 收到的 exclude 应包含其他所有账号 id。"""
    pool, a0, a1 = _pool_with_accounts()

    # 记录 acquire 被调用时的 exclude 参数
    received_excludes = []

    async def _fake_acquire(exclude=None):
        received_excludes.append(exclude)
        return a1

    async def _fake_release(account, success, cooldown=False, **kwargs):
        pass

    pool.acquire = _fake_acquire
    pool.release = _fake_release

    asyncio.run(pool.generate("hi", "gemini-pro", gem_id="g9", account_id="account-1"))

    # 1. 只命中绑定账号 account-1，且带上 gem_id
    assert a1.client.calls == [("generate", "hi", "gemini-pro", "g9")]
    assert a0.client.calls == []

    # 2. 验证锁定逻辑：acquire 收到的 exclude 应包含除 account-1 之外的所有账号（即 account-0）
    assert len(received_excludes) == 1
    assert received_excludes[0] == {"account-0"}


def test_generate_unknown_account_raises():
    """account_id 不存在时应立即抛出 ValueError，不调用 acquire。"""
    pool, a0, a1 = _pool_with_accounts()

    acquire_called = []

    async def _fake_acquire(exclude=None):
        acquire_called.append(True)
        return a0

    pool.acquire = _fake_acquire

    with pytest.raises(ValueError):
        asyncio.run(pool.generate("hi", "gemini-pro", account_id="no-such-account"))

    # acquire 不应被调用
    assert acquire_called == []


def test_generate_stream_pinned_account_passes_gem_id():
    """流式版本：验证 gem_id 透传 + exclude 预填锁定逻辑。"""
    pool, a0, a1 = _pool_with_accounts()

    received_excludes = []

    async def _fake_acquire(exclude=None):
        received_excludes.append(exclude)
        return a1

    async def _fake_release(account, success, cooldown=False, **kwargs):
        pass

    pool.acquire = _fake_acquire
    pool.release = _fake_release

    async def _collect():
        events = []
        async for evt in pool.generate_stream("hi", "gemini-pro", gem_id="g7", account_id="account-1"):
            events.append(evt)
        return events

    events = asyncio.run(_collect())

    # 1. gem_id 透传到目标账号 client
    assert a1.client.calls == [("generate_stream", "hi", "gemini-pro", "g7")]
    assert a0.client.calls == []

    # 2. 收到了流式事件
    assert any(e.get("type") == "delta" for e in events)

    # 3. 锁定逻辑：exclude 包含除 account-1 外的所有账号
    assert len(received_excludes) == 1
    assert received_excludes[0] == {"account-0"}


def test_update_gem_routes_to_account():
    """update_gem 应路由到指定账号的 client。"""
    pool, a0, a1 = _pool_with_accounts()
    result = asyncio.run(pool.update_gem("account-1", "gem-x", "MyGem", "do stuff", "desc"))
    assert result is True
    assert a1.client.calls == [("update_gem", "gem-x", "MyGem", "do stuff", "desc")]
    assert a0.client.calls == []


def test_delete_gem_routes_to_account():
    """delete_gem 应路由到指定账号的 client。"""
    pool, a0, a1 = _pool_with_accounts()
    result = asyncio.run(pool.delete_gem("account-0", "gem-y"))
    assert result is True
    assert a0.client.calls == [("delete_gem", "gem-y")]
    assert a1.client.calls == []
