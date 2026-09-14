# Maintainer tools

| Need | Location |
|---|---|
| Inspect a checkpoint | `checkpoint_inspect.cc` (`flowedge-inspect`) |
| Validate a pinned upstream SmolVLA layout | `smolvla_preflight.py` |
| Construct the upstream SmolVLA source policy | `verification/verify_smolvla_source.py` |
| Export an upstream action chunk for a real capture | `verification/export_smolvla_reference.py` |
| Verify native recognition of a real SmolVLA checkpoint | `verification/verify_smolvla_inspection.py` |
| Check a real GPT-2 conversion against Hugging Face | `verification/verify_transformer_reference.py` |
| Check real GPT-2 token versus external-embedding prefill | `verification/verify_transformer_embeddings.py` |
| Compare or publish benchmarks | `benchmark/` |
| Run model and converter parity checks | `verification/` |

Most users only need the stable commands in `scripts/`. The complete maintainer gate is
`scripts/verify_all.sh`.
