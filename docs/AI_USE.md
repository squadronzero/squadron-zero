# Use of AI in SHIFT

Kept up to date during the build, as the NASA Space Apps 2026 rules ask ("every AI tool
named, the prompts and reasoning behind them, and the data the AI worked on"). The final
wording is aligned with the official Project Submission Guide when it publishes on 13 Nov 2026.

## Tools

| Tool | Used for |
| --- | --- |
| Claude (Anthropic), in the Claude app with code execution and GitHub access | Writing and testing the pipeline code, tests, configuration and documentation in this repository; reading FIRMS files and Colab outputs to check results |
| Gemini (Google), a Gem called "SHIFT Tutor" | Explaining each Colab step to team members and reading error messages. It does not write or change code |
| ChatGPT (OpenAI) | Backup tutor, same role as Gemini |

No AI tool is part of the running product so far. If an in-app explanation agent is added
later, it will only describe numbers that the pipeline computed, and its prompts will be
stored in this repository.

## What the AI did

- Drafted the pipeline modules in `pipeline/shift/` (ingest, aggregate, calibrate, fetch,
  scan, qc, crosscheck, quickstart) and their tests, from the method and decisions the team set.
- Found and fixed data problems by reading the real FIRMS files, for example the VIIRS
  download that used MODIS-style column names, and the Florida box against state outline (D16).
- Wrote the documentation in `docs/` and the step-by-step Colab instructions.

## What the team did

- Chose the challenge, the science question and the method: the v1 Florida prototype and
  its MODIS to VIIRS FRP calibration (VIIRS_FRP = 0.316 x MODIS_FRP + 24.7) are the team's work
  and are the baseline every new result is checked against.
- Made every decision in the decision log (grid size, sensors, years, hosting, scope).
- Requested and downloaded all NASA data, ran every Colab job, and reviewed every result.

## Data the AI worked on

Only public NASA FIRMS active-fire data (MODIS Collection 6.1 and VIIRS 375 m) and the
pipeline's own outputs. No personal data.

## How AI output is checked

- Every number comes from tested code, never from an AI's text. 25 automated tests run on
  GitHub on every change.
- The new pipeline must reproduce the team's v1 Florida results (Phase 1 gate: slope 0.3165
  against v1's 0.3162, 19,476 pairs against 19,689).

## Prompts

The work was done as a conversation, so there is no fixed prompt set. The main recurring
instructions were: "keep the v1 method and results", "Florida v1 is the regression test",
"nothing downstream may depend on a specific data file", and "explain every step in plain
English for the team". Key prompts will be summarised here before submission.
