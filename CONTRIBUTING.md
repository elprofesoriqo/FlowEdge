# Contributing to FlowEdge

Thanks for considering contributing to FlowEdge.

Contributions are welcome across model support, kernels, backends, benchmarking, APIs, documentation, and robotics integrations.

## Where to start

If you're new to the project, look for issues labeled:

* `good first issue` — small, well-scoped tasks suitable for first-time contributors
* `help wanted` — work where community contributions are especially welcome

For large features or architectural changes, please open or comment on an issue first so the scope and interface can be agreed before implementation.

## Development environment

FlowEdge requires:

* CMake 3.21+
* a C++23 compiler

  * Clang 16+
  * GCC 13+
  * MSVC 19.38+

See the `README.md` and project documentation for full build and setup instructions.

A typical build with tests enabled is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Submitting a Pull Request

1. **Fork the repository** and create a branch from `main`.
2. **Keep the PR focused** on one logical change.
3. **Build and run the tests** before submitting.
4. **Follow the existing C++ style** and formatting conventions.
5. **Add or update tests** for changed behavior.
6. **Link the relevant issue** when applicable.
7. **Describe what changed and why** in the PR description.

For performance-sensitive changes such as kernels, scheduling, memory layout, model execution, or backends, include before/after benchmark results and the environment used to produce them.

Changes to the inference hot path should preserve FlowEdge's fixed-memory / allocation-free runtime behavior unless there is a strong reason not to.

## Reporting bugs

Before opening a bug report:

* check that the issue has not already been reported;
* use the bug report template;
* include the FlowEdge version or commit;
* include your OS, CPU/device, compiler, and relevant configuration;
* provide the smallest reproduction you can.

Logs and exact commands are especially helpful.

## Requesting features

Use the feature request template and explain:

* the use case;
* why the feature belongs in FlowEdge;
* the smallest useful scope;
* how the implementation could be validated.

For large features such as new backbones, action heads, or accelerator backends, defining the supported model/hardware scope up front is strongly preferred.

## Performance changes

FlowEdge is designed around deterministic, low-latency inference.

If your change affects performance, please include:

* CPU/device
* OS
* compiler
* build type
* FlowEdge commit
* benchmark configuration
* before/after results

Do not report speedups from runs performed on materially different environments.

## Questions

If you're unsure whether an idea fits the project, open a GitHub Discussion or comment on the relevant issue before starting a large implementation.
