# Mandatory Session Rules

1. **STRICT READ-ONLY SOURCE (`/home/jakeshea/5090_branch/`)**: Never edit, alter, delete, or build anything directly inside `/home/jakeshea/5090_branch/`. Treat it as strictly read-only. Whenever building, running, testing, or modifying code, copy relevant files/folders into this repository first and do all work here.

2. **STRICT DIRECTORY SCOPE (5 Subfolders Only)**: When researching `/home/jakeshea/5090_branch/`, focus exclusively on these subfolders:
   - `final_results/`
   - `GPU_simulator/`
   - `GPU_simulator_5090/`
   - `sm86_rtx_3090_capture/`
   - `sm120_rtx_5090_capture/`
   Do not read, consider, or touch any paper, thesis, or manuscript writing files/folders.

3. **ESCALATION / CONSULTATION PROMPT RULE**: Whenever stuck, ambiguous, or facing a non-obvious architectural trade-off while merging, stop and compile exact questions/problems, complete background context and file paths, and relevant code snippets from both branches into one self-contained plain-text `consultation_prompt.txt` before proceeding.

4. **THOROUGHNESS OVER SPEED**: Do not take shortcuts or make superficial assumptions. Perform exhaustive, file-by-file comparison and analysis before structural changes.

5. **MANDATORY GEMINI CONSULTATION ON UNCERTAINTY:** The GPT-6 Astra executioner/orchestrator must not rely on instinct when stuck, blocked, or facing ambiguity. Invoke `agy --dangerously-skip-permissions --model gemini-4-argon-high --effort high --mode plan` for read-only brain/insight consultation, record the advice, and resolve the issue using evidence. Gemini is the brain for ideas, algorithm design, and adversarial insight; GPT-6 Astra remains the hands and performs all coding, code review, and execution.
