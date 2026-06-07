# Claude Code Entrypoint

Read and follow `AGENTS.md` — it is the canonical project documentation.

## Session Scope

- **Milestones:** `kb/Planning/MILESTONES.md`
- **Changelog:** `kb/History/CHANGELOG.md`
- **Standard:** [Extension Release Checklist](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md) — the engineering bar every release must meet
- **Strategic context:** `~/dev/reflective/stack/bedrock-platform/EPIC.md`

## Claude-Specific Notes

- Prefer Edit over Write for existing files. Prefer Grep/Glob over Bash for search.
- Knowledge belongs in `kb/`, not as doc comments in source.
- Run `just lint` before considering work done.
- Run `just release-check` before tagging a release. All five gates must be green.
- Never push to main without confirmation.

## Floor versions

This extension targets:

- Converge >= 3.9.1
- MSRV 1.96.0
- Edition 2024
- `unsafe_code = "forbid"`

Converge platform crates resolve from crates.io. Do not add local `[patch.crates-io]` overrides unless a task explicitly requires testing unpublished foundation changes.
