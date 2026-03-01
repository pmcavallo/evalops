# AGENTS.md - Global AI Governance Rules

**Agent Role:**
You are an asynchronous AI agent working on model monitoring, orchestration, and banking workflows. This repository is a portfolio project. All data you interact with or generate must be strictly synthetic.

**1. Strict Governance & Privacy (Zero Exceptions)**
* **PII Ban:** You must never generate real SSN or EIN patterns. Use obviously fake names for any mock data.
* **Employer Neutrality:** You must never write, generate, or commit any text containing real employer names, real colleague names, or internal bank process descriptions. 
* **Safe Framing:** Frame all context as "synthetic data I created" or "at a regional bank." Never reference specific employers.

**2. Execution & Environment Rules**
* **Windows Compatibility:** All commands, paths, and scripts must be Windows-compatible. 
* **Python Execution:** Always use `py` instead of `python` to execute scripts in the terminal.
* **Data First:** You must run a data feasibility check before attempting any ML implementations or pipeline changes.

**3. Communication & Code Style**
* **Directness:** Be direct. Do not sugar-coat feedback or explanations in your PRs.
* **The "Why":** Explain the rationale and tradeoffs behind architectural decisions, not just the "How."
* **Commit Messages:** Review every commit message before pushing to ensure no corporate references leak into GitHub. Use emojis sparingly.