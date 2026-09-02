---
description: Verify the FlowEdge C++23 engineering and performance contract.
---

# FlowEdge verification

Read [the engineering standard](../engineering-standard.md), then run:

```bash
FLOWEDGE_BUILD_DIR=build-verify ./scripts/build.sh Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
FLOWEDGE_BUILD_DIR=build-verify ./scripts/lint.sh
ctest --test-dir build-verify --output-on-failure
FLOWEDGE_BUILD_DIR=build-verify ./scripts/verify_all.sh models/mamba_flow.safetensors
```

For a performance change, run its focused benchmark before and after from the same Release build,
CPU power state, affinity policy, and workload. Report the median and keep the raw command. Relay
queue changes must include:

```bash
./build-verify/flowedge_job_queue_bench 3000
./build-verify/flowedge_cooperative_job_bench 1000000
```

Review the diff after automation. Confirm initialization owns every dynamic allocation, runtime
atomics name their memory order, cache-line boundaries isolate contended control state, and public
features have a test, example or installed-consumer check.
