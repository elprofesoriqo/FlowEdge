# Verification

The engine output is checked against PyTorch in CI. Kernels are checked against naive references in unit tests.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  CK[checkpoint] --> CPP["FlowEdge engine"]
  CK --> REF["torch_ref.py"]
  CPP --> D[compare]
  REF --> D
  D --> P{"pass? ULP + rel error"}
```

## The gate

- `scripts/torch_ref.py` builds the golden. It runs the model in PyTorch and dumps the trajectory.
- `scripts/verify_ulp.py` runs the engine on the same input and compares per dimension.
- Two thresholds, both per dimension: relative error below `2e-3` and a ULP distance below `4096`. The build fails on drift.

The gate is deliberately looser than what the engine actually achieves. On the reference checkpoint the flow head lands around 1e-6, so the thresholds leave room for a different host, compiler, or ISA to reorder floating-point work without turning CI red. They catch a real regression, not a last-bit difference.

```bash
python scripts/verify_ulp.py models/mamba_flow.safetensors
```

## Round-trip

The converter is checked too. CI converts a model, then asserts the engine gives identical output from the original and the converted file. A broken mapping fails here.

## Unit tests

`test/tests.cc` compares a kernel to a naive reference written inline, so each
test checks the real kernel against a second implementation. Covered: `matmul`,
`silu`, `softplus`, `rmsnorm`, `conv1d_causal`, `discretize_and_scan`,
`ThreadPool`, and the flow head. Not yet covered: `gate_silu` and
`conv1d_step`.

Add a reference and a test for every new kernel and head.

When Relay is enabled, `RelayProcess.ExchangesRequestWithDaemonAcrossProcessBoundary` launches the
actual `flowedge-relayd` executable. It waits for daemon-owned shared-memory mappings, sends a typed
condition through `RelayClient`, validates the returned action and model identity, then requests a
clean shutdown. This test runs on native Windows and POSIX/WSL builds; component-only ring and worker
tests do not replace it.

`RelayTrace.ReplaysExactConditionAndActionRecords` also asserts representative on-disk bytes for the
v2 file header, record header, envelope integers, and a float payload. This guards against accidentally
reintroducing native-struct serialization while ordinary round-trip tests still pass on one host.
Use `flowedge-relay-trace replay TRACE --model FILE` to compare a field capture against a checkpoint;
the command returns non-zero when a completed recorded action exceeds its configured tolerance.

The process-boundary test also runs the daemon with an intentionally conservative NFE cost. It proves
that one feasible request executes and a tighter request returns a valid `rejected_deadline` action.
Scheduler unit tests cover affected-prefix demand, active remaining work, saturated generation
replacement, capacity displacement feedback, and expiry feedback independently of wall-clock timing.

`HeadWorkerPool.RunsTwoRequestsOnPreallocatedWorkerThreads` dispatches two requests before consuming
either result, validates both actions, and verifies that both fixed slots return to idle. The process
test starts `flowedge-relayd --workers 2`, so lifecycle, admission feedback, and clean shutdown also
exercise the threaded pool rather than a test-only executor.
`HeadWorkerPool.CancelsDispatchedWorkAndInvalidatesBorrowedStaleResults` covers the cross-thread
generation gate and the output-backpressure race where a newer generation arrives while an older
result is borrowed by the transport loop.

## External-head and streaming smoke test

`scripts/verify_external_head.py` generates two small checkpoints without downloading model data.
One contains only a flow head and verifies direct conditions plus bit-identical resumable solving.
The other verifies Mamba batch/streaming parity and decode-state branch restoration through the
Python extension.

```bash
python scripts/verify_external_head.py build
```
