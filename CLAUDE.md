# CLAUDE.md

This file guides Claude Code when working in the Group 16 Personalized Neural
TTS project. Read `AGENTS.md` as the detailed shared repository contract and
`README.md` for the user-facing workflow before changing files.

## Project purpose

This is an academic Speech Processing Assignment 2 implementation. It converts
text to speech with two real Piper voices, speaking-rate control, librosa pitch
post-processing, PCM-16 WAV export, objective signal diagnostics, and genuine
human MOS collection through Streamlit.

The project must remain:

- runnable with Python 3.12 and Pipenv, preferring GPU (CUDA) inference when
  available and always falling back to CPU automatically;
- portable between Windows and the BITS Virtual Lab (Rocky Linux 9.5, no
  NVIDIA driver — `nvidia-smi` is absent there), where the CPU fallback path
  is what actually executes;
- academically honest about measurements, screenshots, and human ratings; and
- submission-ready without hidden absolute paths or undeclared dependencies.

## Start every task here

1. Read `README.md`, `AGENTS.md`, `Pipfile`, and the files in scope.
2. Use `rg`/`rg --files` to inspect references before editing.
3. Use the locked environment; do not install project packages globally.
4. Preserve unrelated user work and generated evidence.
5. Make the smallest coherent change, then run proportional validation.

## Canonical commands

```bash
pipenv sync --dev
pipenv verify
pipenv run check
pipenv run app
pipenv run notebook
```

Only if the four Piper files are missing:

```bash
pipenv run download-models
```

`requirements.txt` is a maintained fallback for environments where Pipenv is
not available. `Pipfile.lock` is the development reproducibility source of truth.

## Code map

- `app.py`: Streamlit presentation, model caching, playback, download, and MOS UI.
- `tts_core.py`: validation, Piper inference, pitch processing, persistence,
  diagnostics, plotting, and reusable domain logic.
- `download_models.py`: official downloader for the female and male model pairs.
- `build_report.py`: source for `Group16_Project_Report_One_Page.pdf`; writes
  `Group16_Project_Report_One_Page.html`, then print that to PDF with headless
  Chrome (see the script's docstring for the exact command).
- `notebooks/Group16_Personalized_TTS.ipynb`: executed academic narrative and experiments.
- `input_data/`: deterministic sample text and blank MOS collection template.
- `outputs/`: generated WAVs, sidecar metrics, plots, and benchmark evidence.
- `models/`: two `.onnx` files and their required adjacent `.onnx.json` files.

## Domain invariants

- `Female` and `Male` map to distinct Piper models; never simulate gender by
  pitch-shifting a single voice.
- UI rate `r` maps to Piper `length_scale = 1 / r`.
- Piper has no native semantic pitch control. Pitch is a transparent librosa
  post-processing step in semitones and must be described that way.
- GPU (CUDA) is the preferred inference target; CPU is the mandatory fallback
  used whenever CUDA is unavailable or a CUDA session fails at run time (e.g.
  BITS Virtual Lab). Never make CUDA a hard requirement — the app must still
  run correctly with `CUDAExecutionProvider` absent.
- WAV output remains mono PCM-16 at the model sample rate unless an assignment
  requirement explicitly changes it.
- Objective diagnostics do not prove naturalness or intelligibility; MOS is a
  separate human evaluation.
- Both the `.onnx` model and adjacent `.onnx.json` config are required.

## Academic-integrity boundaries

Never fabricate or infer:

- MOS ratings, rater counts, or listener comments;
- BITS Virtual Lab execution or screenshots;
- latency, RTF, audio-quality results, or successful test output; or
- official deadlines, portal limits, filename rules, or group members.

The local Streamlit screenshot is not Virtual Lab evidence. Preserve the listed
student identity unless the user explicitly supplies a correction. Do not expose
private rater information or unrelated personal data.

## Editing and artifact rules

- Keep reusable speech logic in `tts_core.py`; keep UI concerns in `app.py`.
- Use relative project paths. `PIPER_MODEL_DIR` is the only supported model-path
  override; never commit a machine-specific `.env`.
- Add type hints and complete docstrings to every new public function or class.
- Explain speech-processing mathematics and non-obvious NumPy/audio operations.
- Keep `Pipfile`, `Pipfile.lock`, and direct constraints in `requirements.txt`
  synchronized when dependencies change.
- Do not hand-edit PDFs, HTML exports, PNGs, WAVs, or JSON metrics. Regenerate
  them from source and measured runs when needed — the one-page report's
  source is `build_report.py`, never edit its `.html`/`.pdf` output directly.
- Do not delete or redownload the large models unless the task requires it.
- Do not initialize Git, publish files, or upload results unless explicitly asked.

## Validation expectations

For documentation or environment-only changes:

```bash
pipenv verify
pipenv run check
pipenv run python -c "from tts_core import validate_text; assert validate_text(' test ') == 'test'"
```

For synthesis changes, additionally run both voices and inspect WAV existence,
sample rate, duration, clipping, and sidecar metadata. For notebook changes, run
all cells from a clean kernel with zero errors before refreshing its PDF/HTML.

## Completion standard

A task is complete only when the requested source and documentation agree, the
relevant Pipenv checks pass, generated claims come from real execution, and any
remaining human-only action is stated plainly.
