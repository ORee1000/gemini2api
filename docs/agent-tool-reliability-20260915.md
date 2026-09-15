# Agent tool reliability review — 2026-09-15

## Scope and evidence

This fork matches the deployed upstream base `6b60269` (v1.6.40). OpenRouter
uses a model API; this project uses Gemini Web and simulates tool calls by
serializing descriptions and history into a prompt, then parsing model text.
A working OpenRouter route does not validate this adapter's protocol handling.

Observed production failures included typed `BardErrorInfo` 1095 / RPC 8 and
1096 / RPC 13. Neither private subcode has a verified specific cause here.
A long real conversation later completed its authored reply, tool call, image
and voice delivery after retries. Do not attribute every failure to context
length, account quota, invalid cookies or a character's decision not to reply.

## Correctness fixes incorporated from the deployed patches

- Empty upstream answers must not complete successfully. HTTP returns an error;
  opened SSE streams carry an error event. Available reasoning remains separate
  from authored output and tool calls.
- Explicit Flash/Flash-lite do not enable extra thinking merely because a client
  supplied generic reasoning effort. Flash-thinking keeps its selected family.
  Buffered tool calls forward the extra-thinking option and available reasoning.
- Historical tool arguments use readable Unicode string tokens on the wire.
  Numbers, duplicate keys, literal code escapes and stored history stay intact.
- A measured Flash Web input budget rejects flattened prompts above 1,000,000
  characters before borrowing an account or sending a request. This is not the
  model API's token context limit, and is not imposed on untested Pro models.
- Typed RPC error frames are parsed before missing-payload frames are skipped.
  RPC 8 is mapped to HTTP 429; unknown typed errors retain numeric diagnostics.
- 429 and partial-stream 5xx do not falsely expire credentials. Actual auth
  failures still follow the existing account-state handling. No new account
  rotation, partial-output replay or success receipt is invented.

## Additional defects found in the source review

1. A required/named tool choice was accompanied by a contradictory plain-text
   alternative and example. That alternative is removed only when the caller
   requires a tool. Tool-disabled requests do not enter tool parsing.
2. The image-intent shortcut could discard an explicitly required tool. Required
   and named tool choices now take precedence; existing automatic image routing
   remains available when tool choice is automatic.
3. Malformed string arguments became `_raw`, non-object arguments became `{}`,
   and bad members of a multi-call list could be silently dropped. Invalid calls
   now invalidate that batch instead of fabricating executable parameters.
4. The OpenAI response path did not validate the selected/declared tool names or
   a required call. Invalid results now return explicit errors rather than a
   successful `stop` or partial executable batch. Valid first-pass calls add no
   model request; existing malformed-JSON regeneration remains bounded to once.
5. Explicit 429/5xx could trigger extra-thinking downgrade or full-history
   replay. These resource failures no longer trigger those adapter-level retries;
   existing configured fallback and caller backoff remain separate.
6. Buffered conversation recovery rebuilt history but omitted the tool schema.
   Recovery now retains the tools and caller's tool-choice constraint.
7. Malformed-call logging included a raw response prefix. It now logs length
   without exposing arguments or private content.

These are protocol and infrastructure fixes. They do not impose a persona,
rewrite authored content, synthesize missing speech/visual metadata, or change
the user's main model/provider selection. The proxy remains a Web adapter;
local validation cannot guarantee every upstream model answer follows a tool
contract or prevent every Google Web refusal.

## Validation and deployment boundaries

- Full Python suite: 681 passed, 30 skipped (Node unavailable), no failures.
  Tests ran as UID 1000 without network or account-data mounts.
- Repository Ruff correctness gate passed.
- Tests cover required/auto/none/named choice, HTTP/SSE, invalid argument batches,
  real account acquisition/release, errors after partial output, RPC frame chunk
  boundaries, one-pass valid tool output, regeneration bounds and disconnects.
- Previous tests for extra thinking now explicitly select Flash-thinking; mock
  client signatures accept the newly forwarded parameter. The old assertion
  that failed malformed-tool regeneration returns a successful 200 is replaced
  with the explicit 502 contract.
- Deployment must keep the current private configuration/data mounts, drain
  active requests, retain a known image/compose rollback, and verify healthy
  startup and source hashes. Do not replay user messages as a deployment test.
- Runtime source now incorporates `fix_response_protocol.py`,
  `fix_web_prompt_transport.py`, `fix_rpc_errors.py`, and
  `fix_pool_resource_errors.py`; do not apply those old hash-checked boot patches
  again to an image built from this branch. Deployment-specific public aliases
  and API authentication remain separately configured.

Local private evidence: `/home/ubuntu/gemini2api-audit-20260915/`.


## Deployment and live verification

The reviewed runtime code (`cb4a98e`) was deployed at 2026-09-15 10:16 UTC in
`local/gemini2api:codex-cb4a98e-r1`. All five affected live source hashes match the
reviewed source plus the existing deployment-specific alias transform. The
native cold-start check passed; gateway drain completed and was cleared. The
main profile remains on OpenRouter; private configuration and persona files
were hash-checked unchanged.

The first local derivative image failed its actual non-root startup because
COPY had left source files owned by root while the existing alias boot script
needs to update them. Deployment automatically rolled back. The derivative
build was corrected with COPY --chown=appuser:appuser and retested through the
real entrypoint before the successful second deployment. This was a packaging
mistake in the local derivative build, not an upstream RPC cause. The repository's
standard Dockerfile already performs the needed ownership setup.

Live probes, with no tool execution or Telegram messages:

- Flash: one neutral request, one tool, HTTP 200, correct transport_probe call
  with ready value and tool_calls completion; 4.33 seconds.
- Flash-thinking: synthetic long history containing 250 messages and 40 tools,
  HTTP 200, correct transport_probe call and tool_calls completion; 40.86 seconds.
- No upstream error frames or successful empty completions in these two probes.

Replay of the original private request dump was blocked by automatic approval
review pending explicit authorization for sending that private context upstream.
That replay did not run. The long probe above uses entirely synthetic records;
it is a successful transport test, not a claim that the original private request
was replayed successfully or that intermittent upstream failures are eliminated.

## Second source review

Additional reproducible defects in OpenAI-compatible delivery:

- Streaming tool calls omitted `index`, required by the OpenAI delta contract.
  Each call now has its own index and id, so consumers can assemble a batch.
  Reference: https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_chunk.py
- A successful non-streamed recovery inside a streaming request could finish
  without emitting its recovered answer. Recovery now emits the answer and
  associated reasoning and retains the requested thinking option.
- Emitted reasoning and final-only text were not consistently counted as output.
  Once either is emitted, failure ends that stream with an error; no regenerated
  or fallback answer is appended. Reasoning without an answer still fails.
- Malformed-tool regeneration swallowed actual upstream exceptions. OpenAI opts
  into error propagation, preserving 429/5xx and terminal recovery errors rather
  than disguising them as generic format failures. Legacy Claude retry handling
  is unchanged; cancellation and the one-regeneration limit remain intact.
- Rejected tool generations were committed before validation, and a successful
  regeneration retained the first attempt's conversation id and reasoning.
  Validation now precedes state changes; accepted text, conversation id and
  reasoning come from the same attempt. Failed attempts do not change the local
  conversation record. This does not undo any remote Gemini conversation state.
- After buffered conversation recovery, malformed-tool regeneration still used
  the expired conversation and short prompt. It now uses the recovered full
  prompt, including tools, with the expired conversation cleared.
- Invalid `tool_calls` could hide behind `status: text`. Explicit malformed tool
  intent now remains a failure even when a text field is also present.

Tests use neutral synthetic content and no tool execution. They cover both HTTP
and SSE, resource errors during regeneration, accepted-versus-rejected history,
partial reasoning, final-only output, recovery, and multiple tool indices.
These defects are proven adapter bugs; they do not establish the private cause
of Google's intermittent 1095/1096 failures or guarantee upstream availability.

Second-review validation: 697 passed, 30 Node-dependent frontend tests skipped,
Ruff correctness gate passed. Sixteen new regression cases exercise the paths
above. Read-only inspection of the deployed Hermes `_ToolCallAccumulator`
confirmed it already tolerates missing indices when ids distinguish calls;
therefore missing indices are not asserted as the sole cause of Kratos failures.
