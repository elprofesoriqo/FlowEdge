# Maintainer tools

| Need | Location |
|---|---|
| Inspect a checkpoint | `checkpoint_inspect.cc` (`flowedge-inspect`) |
| Validate a pinned upstream SmolVLA layout | `smolvla_preflight.py` |
| Construct the upstream SmolVLA source policy | `verification/verify_smolvla_source.py` |
| Export an upstream action chunk for a real capture | `verification/export_smolvla_reference.py` |
| Export upstream VLM K/V and expert velocity for a real capture | `verification/export_smolvla_expert_reference.py` |
| Verify native recognition of a real SmolVLA checkpoint | `verification/verify_smolvla_inspection.py` |
| Check a real GPT-2 conversion against Hugging Face | `verification/verify_transformer_reference.py` |
| Check real GPT-2 token versus external-embedding prefill | `verification/verify_transformer_embeddings.py` |
| Check real SmolVLA action/time suffix projection against PyTorch | `verification/verify_smolvla_action_expert.py` |
| Check a real captured VLM-cache expert replay | `verification/verify_smolvla_cached_expert.py` |
| Export or verify a real cached-VLM Euler trajectory | `verification/export_smolvla_expert_reference.py --trajectory-output` and `verification/verify_smolvla_cached_trajectory.py` |
| Verify source LeRobot preprocessing/VLM-prefix plus native expert action parity | `verification/verify_smolvla_hybrid.py` |
| Measure matched SmolVLA source and hybrid stages | `benchmark/run_smolvla_hybrid_report.py` |
| Compare or publish benchmarks | `benchmark/`; use `benchmark/run_policy_report.py` for the canonical PyTorch report |
| Run model and converter parity checks | `verification/` |

Most users only need the stable commands in `scripts/`. The complete maintainer gate is
`scripts/verify_all.sh`.
