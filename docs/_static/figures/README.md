# Figures

Excalidraw sources (`.excalidraw`) plus GitHub/Sphinx SVGs. GitHub does not render `.excalidraw`.

```bash
python docs/_static/figures/draw.py
```

Also writes `assets/flowedge.gif` (flow-matching ODE) and `assets/perf.gif` (matched PushT vs PyTorch).

`purpose.svg` is Hardware / FlowEdge / LeRobot. `big-flow.svg` is convert → arena → flow or DP → period → motors (no backends; those live on `status.svg`). `relay.svg` is the optional IPC path under that flow. `policies.svg` is the two heads. `why-this.svg` is the README “usual stack” table. Tenstorrent and Metal are one backend (TT-Metal).
