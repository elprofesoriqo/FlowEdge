# Repository and docs map

The public entry points are intentionally small:

| Need | Location |
|---|---|
| Build and run a C++ sample | `README.md`, `examples/` |
| Python API | `python/`, `docs/api/python.md` |
| LeRobot deployment seam | `integrations/lerobot/`, `docs/guides/lerobot.md` |
| Core and Relay contracts | `src/`, `docs/architecture/` |
| Converter and checkpoint checks | `convert/`, `tools/verification/` |
| Reproducible verification | `scripts/verify_all.sh` |

`scripts/` contains user-facing build, test, benchmark, and demo commands. Maintainer-only
benchmark and parity helpers live under `tools/`. Examples and benchmarks stay compiled by
`CMakeLists.txt` and exercised by the verification gate.

`cmake/FlowEdgeConfig.cmake.in` is the install/export template. `CMakeFiles/`, `build*/`,
and `docs/_build/` are generated output and ignored.

## Building the docs

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Open `_build/html/index.html`. CI builds on every pull request and deploys to GitHub Pages on push to `main`. Warnings are errors.
