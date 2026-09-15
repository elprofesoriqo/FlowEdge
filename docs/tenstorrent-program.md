# FlowEdge on Tenstorrent: active program

The first trained robotics target is SmolVLA through LeRobot. This supersedes the
earlier Mamba-first platform sequence; CPU Mamba remains a runtime reference.
Tenstorrent execution is not implemented and no established partnership is claimed.

## Delivery gates

1. Validate visual Diffusion Policy deployment with matched references, replay artifacts,
   and closed-loop simulator results.
2. Pin a trained SmolVLA checkpoint; define exact observation, token, mask, expert, and
   state contracts. The native action/time suffix projection boundary is now covered by
   a real-checkpoint PyTorch parity artifact; capture the remaining intermediate and
   complete-trajectory references next.
3. Implement one fixed-shape SmolVLA path on one Tenstorrent device, with complete Euler
   integration and persistent weights, conditioning, and solver buffers.
4. Measure startup, first/warm inference, host-visible latency, transfers, synchronization,
   memory, trajectory error, and task quality before selecting custom kernel work.
5. Optimize measured bottlenecks with TTNN layouts, buffer reuse, or traces. Contribute
   reusable improvements upstream with minimal reproducers and correctness evidence.
6. Evaluate DeadlineFlow against fixed plans and simpler adaptive baselines, including
   underruns, observation age, deadline misses, task quality, and switching cost.

## Scope

Core owns native inference contracts. Companion adapters own external encoding and
ecosystem integration. The thesis next validates the complete attention expert using
captured encoder outputs. Full-policy claims require actual preprocessing and encoding,
including placement and transfer costs. Accelerator execution needs persistent-device
interfaces, not host pointer substitutions in CPU kernels.

The thesis delivers a correct measured backend. DeadlineFlow is the systems experiment.
CppCon explains measured C++ decisions. Additional hardware generations, policies,
precision portfolios, and custom kernels follow the initial validated slice.

Hardware/cloud access, a technical reviewer, and upstream contribution guidance are the
initial collaboration requests. Tenstorrent's [academic program](https://tenstorrent.com/academic)
offers a public route to discuss them; the roadmap does not imply vendor endorsement.
