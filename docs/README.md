# Repository and docs map

The public entry points are intentionally small:

| Need | Location |
|---|---|
| Build and run a C++ sample | `README.md`, `examples/` |
| Python API | `python/`, `docs/api/python.md` |
| LeRobot deployment seam | `integrations/lerobot/`, `docs/guides/lerobot.md` |
| Core and Relay contracts | `src/`, `docs/architecture/` |
| Converter and checkpoint checks | `convert/`, `scripts/verify_*.py` |
| Reproducible verification | `scripts/verify_all.sh` |

Scripts are grouped by job: `build.sh`, `test.sh`, `lint.sh`, and `sanitize.sh` are
developer wrappers; `verify_all.sh` is the complete gate; benchmark and model-reference
helpers are maintainer tools. Examples and benchmarks are built from `CMakeLists.txt` and
are kept because the verification gate runs them.

`cmake/FlowEdgeConfig.cmake.in` is the install/export template. `CMakeFiles/`, `build*/`,
and `docs/_build/` are generated output and ignored.

## Building the docs

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Open `_build/html/index.html`. CI builds on every pull request and deploys to GitHub Pages on push to `main`. Warnings are errors.
