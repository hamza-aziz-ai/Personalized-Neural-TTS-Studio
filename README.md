# Group 16 - Personalized Neural Text-to-Speech

Course: Speech Processing (S2-25 AIMLCZG520)  
Assignment: Assignment 2, recommended Project 4  
Member: Hamza Aziz (2024AC05133)  
Email: 2024ac05133@wilp.bits-pilani.ac.in

## Primary submission files

- `Pipfile` and `Pipfile.lock` - reproducible Python 3.12 environment
- `Group16_Personalized_TTS_Notebook.pdf` - executed notebook with all cell outputs
- `notebooks/Group16_Personalized_TTS.ipynb` - editable executed notebook
- `Group16_Project_Report_One_Page.pdf` - required one-page report
  (source: `build_report.py`)
- `app.py` - Streamlit user interface
- `tts_core.py` - documented synthesis and evaluation logic
- `input_data/sample_texts.csv` - sample inputs and control settings
- `outputs/*.wav` - four generated neural speech samples
- `screenshots/streamlit_app_result.png` - verified local result screenshot

## Setup and execution

The reproducible development environment uses Python 3.12 and Pipenv. It was
tested with Python 3.12.13 and Pipenv 2026.6.2. From this directory:

```bash
python -m pip install --user pipenv
pipenv sync --dev
pipenv run check
pipenv run app
```

Useful project commands:

```bash
pipenv run notebook          # Open notebooks/Group16_Personalized_TTS.ipynb in JupyterLab
pipenv run download-models   # Only when the four local model files are missing
pipenv verify                # Confirm Pipfile and Pipfile.lock are synchronized
```

`pipenv sync --dev` installs the exact locked runtime and notebook environment.
For a restricted Virtual Lab where Pipenv is unavailable, use the maintained
fallback instead:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

`download_models.py` obtains two official Piper voice/config pairs (about 126 MB):

- `en_US-hfc_female-medium`
- `en_US-hfc_male-medium`

The application prefers GPU (CUDA) inference when a working NVIDIA CUDA/cuDNN
install is detected, and automatically falls back to CPU otherwise. The BITS
Virtual Lab runs Rocky Linux 9.5 with no NVIDIA driver (`nvidia-smi` is
unavailable there), so the CPU fallback path is what actually runs in the
lab; the same code and Pipfile work unchanged on both platforms. Either way
it produces lossless PCM-16 WAV audio without FFmpeg. The relocated project
already contains all four voice files in `models`. The lean ZIP omits the
large models, while the offline ZIP includes them.

`PIPER_MODEL_DIR` may point to another model directory. Copy `.env.example` to
`.env` and set the path there if an override is required; `.env` is intentionally
ignored because machine-specific paths and secrets must not be committed.

## Development guidance

- `CLAUDE.md` contains Claude Code-specific operating guidance.
- `AGENTS.md` is the repository contract for ChatGPT Codex and other coding agents.
- `Pipfile.lock` is the reproducibility source of truth. Keep `requirements.txt`
  aligned as the portable fallback when direct dependencies change.
- Exported PDFs, images, WAV files, metrics, and ZIPs are generated evidence;
  never hand-edit them or invent MOS/Virtual Lab results.

## Evaluation evidence

- Four configured samples were generated at 22,050 Hz.
- Final clean run: 15/15 code cells executed, zero errors.
- Seven warm runs: median latency 0.095 s, p95 0.152 s, median RTF 0.021
  (measured on GPU; CPU-only runs, e.g. the Virtual Lab, are slower).
- Four invalid-input tests passed.
- `outputs/benchmark_results.csv` and `outputs/latency_summary.json` contain exact values.
- The Streamlit form appends genuine listener ratings to `outputs/mos_feedback.csv`.

No MOS value has been fabricated. Collect at least five genuine ratings and rerun
the MOS notebook cell before claiming an average.

## Required user-only evidence

The official rubric awards two marks specifically for BITS Virtual Lab use. The
included local screenshot is not Virtual Lab evidence. Follow
`screenshots/BITS_VIRTUAL_LAB_INSTRUCTIONS.md`, then save the authentic result as
`screenshots/bits_virtual_lab_result.png`.

The official PDF and course handout provide no filename convention, portal upload
limit, or Virtual Lab procedure. Check the current Elearn/Canvas announcement
before upload.

## Model license and responsible use

The Hi-Fi Captain voice model cards identify the data license as CC BY-NC-SA 4.0.
This is a non-commercial academic prototype. Outputs are synthetic and should not
be used for impersonation. Human MOS responses should contain no personal data.
