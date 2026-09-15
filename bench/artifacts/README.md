# Benchmark artifacts

Artifacts are grouped by the claim they support:

- `policy/` — full-policy replay and LeRobot/PyTorch comparison reports.
- `transformer/` — real Transformer checkpoint parity and embedding reports.
- `smolvla/` — native inspection, preflight, suffix parity, and real captured-cache expert/Euler replay; these do not claim native full SmolVLA inference.

Use `tools/benchmark/run_policy_report.py` for a new matched policy report so
the JSON and Markdown files are generated together and validated before review.
