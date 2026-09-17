# Repository and docs map

![convert, load arena, integrate ODE, period, act](_static/figures/workflow.svg)

| Need | Location |
|---|---|
| Build and run a sample | `README.md`, `examples/` |
| Python API | `python/flowedge_ext.cc`, `docs/api/python.md` |
| LeRobot plugin / hardware seam | `integrations/lerobot/`, `docs/guides/lerobot.md` |
| Core and Relay | `src/`, `docs/architecture/` |
| Convert, bench, verify, rollout | `python -m flowedge_dev` (`flowedge_dev/`, `convert/`, `tools/`) |
| Local Grok Bot gate | `scripts/verify_all.sh` / `scripts/verify_local.ps1` |

`scripts/` is user-facing. `flowedge_dev` is the maintainer front door. CMake still builds examples and C++ benches.

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Open `_build/html/index.html`. CI treats warnings as errors.
