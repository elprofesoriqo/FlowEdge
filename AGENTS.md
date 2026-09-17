# Agent notes

Verification is local. Grok Bot runs it on this machine (`/verify-local`).
GitHub Actions compiles and runs `ctest`; it does not own PyTorch ULP, convert
round-trip, or diffusion parity.

- Full gate: `scripts/verify_all.sh` (Linux/Git Bash) or `scripts/verify_local.ps1` (Windows)
- Skill: `.cursor/skills/verify-local/SKILL.md`
- Contract: `docs/guides/verification.md`
