"""Exercise real account acquisition/release with synthetic clients, no network."""
import importlib.util
from pathlib import Path
import sys
import unittest

from app.core.gemini_client import HTTPStatusError, _raise_wrb_error
from app.core.account_pool import Account, AccountPool, AccountStatus, _error_summary


class Client:
    is_healthy=True
    def __init__(self,status=429,partial=False):
        self.status=status;self.partial=partial;self.calls=0

    def error(self):
        e=HTTPStatusError(self.status,'private credential must not appear')
        e.upstream_rpc_status=8 if self.status==429 else 13
        e.upstream_bard_code=1095 if self.status==429 else 1096
        return e

    async def generate(self,*args,**kwargs):
        self.calls+=1
        if self.status:raise self.error()
        return {'text':'Ready.','images':[],'thoughts':''}

    async def generate_stream(self,*args,**kwargs):
        self.calls+=1
        if self.partial:yield {'type':'delta','text':'Partial'}
        if self.status:raise self.error()
        yield {'type':'final','text':'Ready.','images':[],'thoughts':''}


def fixture(status=429,partial=False):
    client=Client(status,partial)
    account=Account('fixture','not-a-cookie','not-a-cookie',client=client)
    pool=AccountPool();pool._accounts=[account]
    return pool,account,client


class PoolTest(unittest.IsolatedAsyncioTestCase):
    async def call(self,pool,stream):
        if stream:
            return [event async for event in pool.generate_stream('Readiness check.','gemini-flash-thinking')]
        return await pool.generate('Readiness check.','gemini-flash-thinking')

    async def test_repeated_429_does_not_expire_or_reset_account(self):
        for stream,partial in ((False,False),(True,False),(True,True)):
            pool,account,client=fixture(partial=partial)
            account.consecutive_failures=2
            for _ in range(4):
                with self.assertRaises(HTTPStatusError) as caught:
                    await self.call(pool,stream)
                self.assertEqual(caught.exception.status_code,429)
                self.assertEqual(account.status,AccountStatus.ACTIVE)
                self.assertEqual(account.consecutive_failures,2)
                self.assertEqual(account.active_requests,0)
                self.assertIsNone(account.last_success_at)
            self.assertEqual(client.calls,4)  # No new retry/failover added.
            self.assertEqual(account.error_count,4)
            self.assertIn('BardErrorInfo 1095',account.last_error)
            self.assertNotIn('private',account.last_error)
            client.status=0
            await self.call(pool,stream)
            self.assertEqual(account.consecutive_failures,0)
            self.assertIsNotNone(account.last_success_at)

    async def test_partial_5xx_preserves_account_and_never_replays(self):
        pool,account,client=fixture(status=502,partial=True)
        for _ in range(3):
            with self.assertRaises(HTTPStatusError):await self.call(pool,True)
        self.assertEqual(account.status,AccountStatus.ACTIVE)
        self.assertEqual(account.active_requests,0)
        self.assertEqual(client.calls,3)

    async def test_auth_failure_still_expires(self):
        for stream in (False,True):
            pool,account,client=fixture(status=401)
            with self.assertRaises(HTTPStatusError):await self.call(pool,stream)
            self.assertEqual(account.status,AccountStatus.EXPIRED)
            self.assertEqual(account.active_requests,0)

    async def test_oversize_never_borrows_an_account(self):
        pool,account,client=fixture()
        for stream in (False,True):
            with self.assertRaises(HTTPStatusError):
                if stream:
                    async for _ in pool.generate_stream('x'*1_000_001,'gemini-flash'):pass
                else:await pool.generate('x'*1_000_001,'gemini-flash')
        self.assertEqual(client.calls,0)
        self.assertEqual(account.request_count,0)
        self.assertEqual(account.error_count,0)

    def test_real_rpc_metadata_is_safe(self):
        frame=['wrb.fr',None,None,None,None,[8,'private',[['type.googleapis.com/assistant.boq.bard.application.BardErrorInfo',[1095]]]]]
        with self.assertRaises(HTTPStatusError) as caught:_raise_wrb_error(frame)
        self.assertEqual(_error_summary(caught.exception),'HTTPStatusError (HTTP 429); RPC 8; BardErrorInfo 1095')
        error=HTTPStatusError(429,'private');error.upstream_bard_code='<script>private</script>'
        self.assertEqual(_error_summary(error),'HTTPStatusError (HTTP 429)')


if __name__=='__main__':unittest.main()
