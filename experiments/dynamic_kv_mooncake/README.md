# Mooncake vLLM forward-port status

This branch is based on vLLM upstream commit `dc9f845` and includes the
backend-neutral connector-preserving sleep commit `8fe5ea9`.

The files in this directory are the exact patchers and tests used by the
accepted Mooncake A/B/C runtime. They remain the authority for the Mooncake
connector behavior:

- save KV for dynamic `FINISHED_ABORTED` requests;
- retain delayed-free blocks until the final save completes;
- preserve the connector across dynamic hybrid sleep;
- emit strict scheduler/worker transfer evidence;
- support the tested TCP CPU-staging path.

## Forward-port TODO

The accepted runtime came from `verlai/verl:vllm023.dev1`. Its prepared
Mooncake scheduler already contained prerequisite tracing and lookup changes.
The current `dc9f845` scheduler has evolved, so
`apply_vllm_mooncake_abort_reuse.py` cannot be replayed mechanically against
this checkout (`LOOKUP_FORCED_MISS` context mismatch).

Before treating this branch as runnable Mooncake code:

1. Port the three patchers semantically to the current scheduler/worker APIs.
2. Run the included signature and aborted-final-save tests in the target image.
3. Re-run at least the strict cross-node C precheck.

