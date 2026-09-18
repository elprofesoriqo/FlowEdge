# Product Direction

FlowEdge is the boundary between a *trained* policy and *robot* control: convert once, fixed RSS, a period, a replayable action. Hardware runs the loop. FlowEdge runs the native head. LeRobot (or your stack) trains, encodes observations, and owns e-stop.

```{image} _static/figures/purpose.svg
:alt: Hardware, FlowEdge, LeRobot
:class: fe-fig
```

| Keep | Leave |
|---|---|
| Hardware: CPU now; Jetson/ARM via plugin; CUDA Mamba, flow, and DP heads; matched GPU replay ([#163](https://github.com/elprofesoriqo/FlowEdge/issues/163)) and CUDA Core period loop ([#164](https://github.com/elprofesoriqo/FlowEdge/issues/164)) on GTX 1650 | Robot e-stop and joint limits |
| FlowEdge: flow matching, DP U-Net, Transformer decoder fixture, optional Relay | Training, datasets, graph compilers |
| LeRobot: plugin, encoders, drivers | Native kernels and the arena |

The sequence is convert → matched CPU replay vs PyTorch → Transformer fixture (#10) → CUDA device-resident flow ([#161](https://github.com/elprofesoriqo/FlowEdge/issues/161)) and DP ([#162](https://github.com/elprofesoriqo/FlowEdge/issues/162)) → matched GPU replay ([#163](https://github.com/elprofesoriqo/FlowEdge/issues/163)) → a period loop on Jetson or SO-100 with miss counts. Tenstorrent stays behind that GPU evidence. Success is an external team doing that without FlowEdge training code.

```{image} _static/figures/sequence.svg
:alt: convert, CPU replay, Jetson or SO-100, same contracts
:class: fe-fig
```
