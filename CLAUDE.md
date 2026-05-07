# CLAUDE.md

This repository uses AGENTS.md as the canonical coding-agent instruction file.

**MANDATORY: At the start of every session, read AGENTS.md in full before doing anything else — including answering questions, exploring code, or running commands.**

Notes:
- this project uses uv. don't mess with .venv or the system python.
- **DO NOT create new git branches unless the user explicitly asks.** Work
  on the currently checked-out branch (default `main`). See AGENTS.md
  "Git Branching Policy" for the full rule. This applies to all
  subagents you spawn — pass the policy through.