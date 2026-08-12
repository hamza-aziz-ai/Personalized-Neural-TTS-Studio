# AGENTS.md

## Scope

These instructions apply to this project root and every descendant directory.
A more specific `AGENTS.md` in a subdirectory overrides this file for that
subtree. This repository contains academic source code and generated submission
evidence, so changes must preserve reproducibility and provenance.

## Mission

Maintain the Group 16 Personalized Neural Text-to-Speech system for BITS Pilani
WILP Speech Processing (S2-25 AIMLCZG520), Assignment 2, Project 4. Optimize for:

1. correct and readable speech-processing implementation;
2. a clean, top-to-bottom executable notebook;
3. GPU-preferred, CPU-fallback portability (Windows dev machine and the
   GPU-less Rocky Linux 9.5 BITS Virtual Lab) and locked dependencies;
4. truthful evaluation and academic integrity.

## Read before editing

Inspect these sources in order, limited to the task at hand:

1. `README.md` for supported workflows and known user-only actions.
2. `Pipfile` and `Pipfile.lock` for the canonical environment.
3. `tts_core.py` for domain behavior and `app.py` for Streamlit behavior.
4. `notebooks/Group16_Personalized_TTS.ipynb` for assignment narrative and experiments.
5. `SUBMISSION_CHECKLIST.md` before changing any release artifact.

Use `rg` or `rg --files` for discovery. Do not assume a Git repository exists,
and do not initialize one unless the user asks.

## Environment and commands

Required interpreter: Python 3.12. The project is tested on Python 3.12.13.
Pipenv is the primary workflow:

```bash
pipenv sync --dev
pipenv verify
pipenv run check
pipenv run app
pipenv run notebook
```

Download models only when missing:

```bash
pipenv run download-models
```

Fallback for a restricted lab without Pipenv:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Do not use global package installation for normal development. If a dependency
changes, update the `Pipfile`, regenerate `Pipfile.lock`, and keep the direct
fallback constraints in `requirements.txt` aligned.

## Architecture and ownership

| Path | Responsibility | Editing rule |
| --- | --- | --- |
| `tts_core.py` | Reusable validation, synthesis, DSP, metrics, saving | Domain logic belongs here; retain type hints and docstrings. |
| `app.py` | Streamlit UI, cache, playback, download, MOS form | Keep UI state and presentation here; call core functions. |
| `download_models.py` | Official Piper model acquisition | Keep idempotent and use `sys.executable`. |
| `build_report.py` | Source for `Group16_Project_Report_One_Page.pdf` | Edit this, then regenerate the `.html`/`.pdf`; never hand-edit either output. |
| `notebooks/Group16_Personalized_TTS.ipynb` | Executed academic explanation and experiments | Markdown before important code; no hidden state. |
| `input_data/` | Sample text and blank human-rating template | Do not invent completed ratings. |
| `models/` | Two Piper ONNX/config pairs | Large licensed assets; do not alter or delete casually. |
| `outputs/` | Measured audio, metrics, and figures | Regenerate from code; never hand-edit evidence. |

## Speech-processing invariants

- The two UI voice choices must resolve to two distinct model IDs:
  `en_US-hfc_female-medium` and `en_US-hfc_male-medium`.
- Every Piper `.onnx` requires its adjacent `.onnx.json` configuration.
- Speaking rate uses `length_scale = 1 / rate_multiplier`.
- Pitch is optional librosa post-processing in semitones, not a native Piper
  capability and not a substitute for a genuinely different voice.
- Execution prefers CUDA (GPU) when a working `CUDAExecutionProvider` is
  present and automatically falls back to CPU otherwise; CUDA must never be a
  hard requirement, and FFmpeg is not required either way.
- Generated WAVs are mono PCM-16 and retain the Piper sample rate.
- RTF is generation latency divided by audio duration. Non-reference signal
  diagnostics must not be presented as PESQ, STOI, SNR, or human naturalness.

## Academic and evidence rules

The following require real user or environment evidence and cannot be generated
by an agent:

- listener MOS scores, sample size, comments, or consent;
- a screenshot proving execution in the BITS Virtual Lab; and
- portal deadlines, upload limits, or naming rules absent from official files.

Do not relabel `screenshots/streamlit_app_result.png` as Virtual Lab evidence.
Do not claim a MOS average until genuine responses exist. Preserve the recorded
student name, ID, group, and course metadata unless the user supplies a change.

## Change workflow

1. Inspect the relevant source, references, and current generated artifacts.
2. State any assumption that materially affects implementation.
3. Make focused edits; preserve unrelated files and user-generated evidence.
4. Run the smallest validation set that can falsify the change.
5. If behavior or measured values changed, regenerate affected notebook outputs,
   figures, audio, report text, and release artifacts from real execution.
6. Report what changed, what passed, and what still needs genuine user evidence.

Use relative paths throughout code and documentation. The supported external
model override is `PIPER_MODEL_DIR`; keep actual `.env` files, secrets, and local
absolute paths out of committed or submitted material.

## Validation matrix

Environment or documentation change:

```bash
pipenv verify
pipenv run check
pipenv run python -c "from tts_core import validate_text; assert validate_text(' test ') == 'test'"
```

Core synthesis change:

- exercise both voice labels;
- check blank text, invalid voice, and out-of-range controls;
- confirm WAV and JSON output creation;
- check sample rate, duration, peak/RMS, clipping, and RTF; and
- compare claims only with newly measured results.

Notebook change:

- execute from a clean kernel, top to bottom;
- require all 15 code cells to complete with no traceback unless the notebook
  structure was intentionally changed;
- preserve visible outputs needed by the assignment; and
- refresh PDF/HTML only from the executed notebook.

## Artifact safety

Treat PDFs, HTML exports, PNGs, WAVs, and JSON metrics as generated outputs.
Do not edit their contents manually. Do not delete, rename, move, or
overwrite evidence outside the requested scope. Before regenerating one,
verify exact target paths and preserve the source files.

## Coding standards

- Prefer clear Python over compressed one-liners.
- Give every new function/class a useful docstring and type hints.
- Explain non-obvious DSP mathematics, array shapes, clipping, and file formats.
- Validate user input and raise actionable errors.
- Keep deterministic seeds where experiments depend on randomness.
- Close Matplotlib figures and avoid retaining unnecessary audio arrays.
- Keep Streamlit startup non-blocking in automated validation.

## Definition of done

The requested behavior and documentation agree; Pipenv is locked and verified;
proportional tests pass; no measured or human evidence was fabricated; and
remaining MOS/Virtual Lab actions are clearly identified as user-only.
