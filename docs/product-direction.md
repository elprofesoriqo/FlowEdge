# Product Direction

FlowEdge is the boundary between a *trained* policy and *robot* control: convert once, fixed RSS, a period, a replayable action. Hardware runs the loop. FlowEdge runs the native head. LeRobot (or your stack) trains, encodes observations, and owns e-stop.

```{image} _static/figures/purpose.svg
:alt: Hardware, FlowEdge, LeRobot
:class: fe-fig
```

| Keep | Leave |
|---|---|
| Hardware: CPU now; Jetson/ARM via plugin; accelerators planned | Robot e-stop and joint limits |
| FlowEdge: flow matching, DP U-Net, optional Relay | Training, datasets, graph compilers |
| LeRobot: plugin, encoders, drivers | Native kernels and the arena |

The sequence is convert → matched CPU replay vs PyTorch → a period loop on Jetson or SO-100 with miss counts → the same contracts on an accelerator. Success is an external team doing that without FlowEdge training code.

```{image} _static/figures/sequence.svg
:alt: convert, CPU replay, Jetson or SO-100, same contracts
:class: fe-fig
```
