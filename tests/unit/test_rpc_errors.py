"""Run in the pinned proxy image with a candidate source root; no live requests."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from app.core.gemini_client import (
    GeminiWebClient, HTTPStatusError, _extract_text_from_wrb,
    _raise_wrb_error, _scan_complete_wrb_frames,
)
from app.routers import openai as api


def error_frame(code=1095, rpc=8):
    return ['wrb.fr', None, None, None, None, [rpc, 'private upstream text', [
        ['type.googleapis.com/assistant.boq.bard.application.BardErrorInfo', [code]],
    ]]]


class RpcTest(unittest.IsolatedAsyncioTestCase):
    def test_buffered_and_chunk_boundaries_surface_error(self):
        frame = error_frame()
        wire = ")]}'\n" + json.dumps([frame]) + '\n'
        client = object.__new__(GeminiWebClient)
        with self.assertRaises(HTTPStatusError) as caught:
            client._parse_output(wire)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertIn('BardErrorInfo 1095', str(caught.exception))
        self.assertNotIn('private', str(caught.exception))
        for cut in range(len(wire)+1):
            buf = ''
            seen = []
            for chunk in (wire[:cut], wire[cut:]):
                buf += chunk
                frames, consumed = _scan_complete_wrb_frames(buf)
                buf = buf[consumed:]
                seen.extend(frames)
            self.assertEqual(seen, [frame])
            with self.assertRaises(HTTPStatusError):
                _extract_text_from_wrb(seen[0])

    def test_normal_content_queue_and_unknown_errors(self):
        payload = [None, None, None, None, [[None, ['Ready.']]]]
        frame = ['wrb.fr', None, json.dumps(payload)]
        self.assertEqual(_extract_text_from_wrb(frame)[0], 'Ready.')
        for other in (None, [], ['wrb.fr'], ['wrb.fr', None, None, None, None, [8]],
                      ['wrb.fr', None, None, None, None, [8, None, []]],
                      ['other', None, None, None, None, error_frame()[5]],
                      error_frame(code=0), error_frame(code='1095')):
            _raise_wrb_error(other)
        with self.assertRaises(HTTPStatusError) as caught:
            _raise_wrb_error(error_frame(9999, 13))
        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn('9999', str(caught.exception))
        self.assertNotIn('context', str(caught.exception))
        # Partial content must not turn a subsequent upstream error into success.
        with self.assertRaises(HTTPStatusError):
            object.__new__(GeminiWebClient)._parse_output(json.dumps([frame, error_frame()]))

    async def test_openai_error_envelopes(self):
        app = FastAPI()
        app.state.model_mapping = NS(resolve=lambda model: model)
        app.include_router(api.router)
        api.limiter.enabled = False

        async def generate(*args, **kwargs):
            return object.__new__(GeminiWebClient)._parse_output(json.dumps([error_frame()]))

        async def generate_stream(*args, **kwargs):
            _extract_text_from_wrb(error_frame())
            if False:
                yield None

        async def fallback(*args, **kwargs):
            return None

        async def stream_fallback(*args, **kwargs):
            if False:
                yield None

        cases = 0
        for model in ('gemini-flash', 'gemini-flash-thinking'):
            for stream in (False, True):
                for tools in (False, True):
                    request = {'model': model, 'stream': stream, 'messages': [{'role': 'user', 'content': 'Readiness check.'}]}
                    if tools:
                        request['tools'] = [{'type': 'function', 'function': {'name': 'record_test', 'parameters': {'type': 'object', 'properties': {}}}}]
                    with patch.object(api.gemini_client, 'generate', generate), patch.object(api.gemini_client, 'generate_stream', generate_stream), patch.object(api, '_fallback_result', fallback), patch.object(api, '_maybe_fallback_stream', stream_fallback):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
                            response = await client.post('/chat/completions', json=request)
                    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]'] if stream else [response.json()]
                    errors = [frame['error'] for frame in frames if 'error' in frame]
                    self.assertEqual(len(errors), 1)
                    self.assertIn('1095', errors[0]['message'])
                    self.assertNotIn('private', response.text)
                    for frame in frames:
                        for choice in frame.get('choices', []):
                            self.assertNotIn(choice.get('finish_reason'), ('stop', 'tool_calls'))
                            delta = choice.get('delta', {})
                            self.assertFalse(delta.get('content') or delta.get('tool_calls'))
                    if not stream:
                        self.assertEqual(response.status_code, 429)
                    cases += 1
        print(json.dumps({'rpc_asgi_cases': cases, 'external_calls': 0}))


if __name__ == '__main__':
    unittest.main()
