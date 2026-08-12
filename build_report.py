"""Build the one-page project report from real measured data.

This is the maintained source for ``Group16_Project_Report_One_Page.pdf``.
Running it regenerates ``Group16_Project_Report_One_Page.html`` (self-contained,
with images inlined as base64); print that file to PDF with headless Chrome:

    python build_report.py
    "<path to chrome>" --headless --disable-gpu --no-pdf-header-footer \\
        --print-to-pdf="Group16_Project_Report_One_Page.pdf" \\
        "file:///<absolute path to Group16_Project_Report_One_Page.html>"

Course: Speech Processing (S2-25 AIMLCZG520)
Group: Group 16
Author: Hamza Aziz (2024AC05133)
"""

from __future__ import annotations

import base64
from pathlib import Path

REPO = Path(__file__).resolve().parent


def b64(path: str) -> str:
    """Read a repo-relative file and return its base64-encoded contents.

    Args:
        path: Repo-relative file path.

    Returns:
        ASCII base64 string suitable for a data: URI.
    """

    return base64.b64encode((REPO / path).read_bytes()).decode("ascii")


def build_html() -> str:
    """Render the one-page report as self-contained HTML.

    Returns:
        Complete HTML document as a string.
    """

    shot_app = b64("screenshots/streamlit_app_result_report_crop.png")
    shot_wave = b64("outputs/waveform_mel_spectrogram.png")
    shot_perf = b64("outputs/performance_comparison.png")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Group16_Project_Report_One_Page</title>
<style>
  @page {{ size: A4; margin: 10mm 11mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.4pt;
    line-height: 1.32;
    color: #17242E;
    margin: 0;
  }}
  h1 {{ font-size: 14pt; margin: 0; color: #123B5D; }}
  .subhead {{ font-size: 8.6pt; color: #4A5B66; margin: 1mm 0 3mm; }}
  h2 {{
    font-size: 8.8pt; text-transform: uppercase; letter-spacing: 0.02em;
    color: #123B5D; border-bottom: 1px solid #D5E4EA;
    margin: 2.4mm 0 1.2mm; padding-bottom: 0.6mm;
  }}
  p {{ margin: 0 0 1.6mm; }}
  ol {{ margin: 0 0 1.6mm 3.6mm; padding: 0; }}
  li {{ margin-bottom: 0.6mm; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 6mm; }}
  .flow {{ display: flex; gap: 1mm; margin: 1mm 0 2.4mm; flex-wrap: wrap; }}
  .flow div {{
    background: #F4F8FA; border: 1px solid #D5E4EA; border-radius: 2mm;
    padding: 1mm 1.6mm; font-size: 7.2pt; text-align: center; flex: 1 1 auto;
  }}
  .flow span {{ align-self: center; color: #97A6AF; font-size: 8pt; }}
  .shots {{ display: flex; gap: 2mm; margin-top: 1mm; }}
  .shots figure {{ margin: 0; flex: 1; text-align: center; }}
  .shots img {{ width: 100%; border: 1px solid #D5E4EA; border-radius: 1.2mm; display: block; }}
  .shots figcaption {{ font-size: 6.6pt; color: #4A5B66; margin-top: 0.6mm; }}
  .foot {{
    margin-top: 2.6mm; padding-top: 1.6mm; border-top: 1px solid #D5E4EA;
    font-size: 7.2pt; color: #4A5B66;
  }}
  .foot .warn {{ color: #A33; font-weight: 600; }}
</style>
</head>
<body>

<h1>Assignment 2 &mdash; Personalized Text-to-Speech System</h1>
<div class="subhead">Speech Processing (S2-25 AIMLCZG520) | Project 4 (recommended) | Group 16 | Hamza Aziz (2024AC05133)</div>

<h2>Objective</h2>
<p>Build an application that converts user text into natural-sounding speech using genuine male/female
neural voices, independent rate and pitch controls, in-app playback and WAV export. Inference prefers
GPU (CUDA) execution and falls back to CPU automatically, so the same code runs unchanged on a Windows
workstation and on the GPU-less Rocky Linux 9.5 BITS Virtual Lab. Evaluate response time, objective
signal diagnostics, and genuine listener MOS.</p>

<div class="cols">
<div>

<h2>System architecture / workflow</h2>
<div class="flow">
  <div>Text input<br>+ validation</div><span>&rarr;</span>
  <div>eSpeak-ng<br>phonemes</div><span>&rarr;</span>
  <div>Piper VITS ONNX<br>(GPU, CPU fallback)</div><span>&rarr;</span>
  <div>Rate + pitch<br>processing</div><span>&rarr;</span>
  <div>PCM-16 WAV<br>play / save</div><span>&rarr;</span>
  <div>Metrics<br>+ MOS</div>
</div>

<h2>Tools, libraries, and technologies used</h2>
<p>Python 3.14.6; Piper TTS 1.6.0; Streamlit 1.61.1; librosa 1.0.0; SoundFile
0.14.0; NumPy 2.5.2; Matplotlib 3.11.1; ONNX Runtime GPU 1.28.0 (CUDA execution
provider preferred, CPU provider as automatic fallback). Voices:
en_US-hfc_female-medium and en_US-hfc_male-medium (22,050&nbsp;Hz; CC BY-NC-SA 4.0).</p>

<h2>Implementation details and key features</h2>
<ol>
  <li>Input normalization, empty/length/range validation, and actionable
  missing-model errors; character limit is UI-configurable (default 5,000).</li>
  <li>Two authentic single-speaker models; gender is not simulated by pitch.</li>
  <li>Rate multiplier maps to length_scale&nbsp;=&nbsp;1/rate; pitch is transparently
  post-processed in semitones because Piper has no native pitch field.</li>
  <li>GPU (CUDA) preferred with an automatic, self-tested fallback to CPU. Sentence
  audio chunks stream in with a live progress bar, are concatenated, clipped safely,
  encoded as PCM-16 WAV, played, downloaded, and saved with JSON metadata.</li>
  <li>Innovation: cached ONNX loading, waveform/log-mel dashboard, RTF and
  quality diagnostics, streaming synthesis progress, listener MOS capture, and
  reproducible negative tests.</li>
</ol>

<h2>Challenges faced</h2>
<p>Voice models total about 126&nbsp;MB and introduce a cold start; model loading
is cached and measured separately. The first pitch-shift kernel has a one-time
compile cost, so explicit warm-up prevents biased steady-state timing. Enabling
GPU execution on Windows required exact-matched cuDNN/cuBLAS wheel versions
(onnxruntime-gpu&nbsp;1.28 + cuDNN&nbsp;9.8.0.87 + cuBLAS&nbsp;12.8.4.1); a mismatch fails
silently mid-inference, so voice loading now runs a real warm-up synthesis and
falls back to CPU on any failure. A PDF cannot carry playable audio, so four WAV
files accompany the notebook. Extreme rate/pitch settings can reduce naturalness.</p>

</div>
<div>

<h2>Results and observations</h2>
<p>Four configured samples (GPU): mean synthesis response&nbsp;=&nbsp;0.165&nbsp;s; mean
RTF&nbsp;=&nbsp;0.036; maximum clipping&nbsp;=&nbsp;0.00097%; all outputs&nbsp;=&nbsp;22,050&nbsp;Hz WAV.</p>
<p>Seven repeated warm runs: median latency&nbsp;=&nbsp;0.094&nbsp;s; p95&nbsp;=&nbsp;0.139&nbsp;s; median
RTF&nbsp;=&nbsp;0.022. RTF below 1 means generation is faster than playback.</p>
<p>Cold model loads: female&nbsp;=&nbsp;2.63&nbsp;s; male&nbsp;=&nbsp;2.02&nbsp;s.</p>
<p>Moderate rate and pitch changes produced distinct durations/spectral profiles
without material clipping. Four invalid-input tests and all 15 notebook code
cells passed with zero errors.</p>

<h2>Evaluation status and responsible use</h2>
<p><b>MOS:</b> 5 genuine self-collected ratings via the Streamlit form (Male voice,
rate 1.00&times;, pitch +0.0&nbsp;st): mean naturalness&nbsp;=&nbsp;clarity&nbsp;=&nbsp;overall&nbsp;=&nbsp;4.0/5
(std&nbsp;0.71, n=5). No rating was fabricated. Coverage is limited to this one
voice/setting &mdash; the female voice and other rate/pitch combinations are not
yet rated. Objective RMS, peak, clipping, silence, centroid, and zero-crossing
values are diagnostics, not substitutes for perceptual judgment.</p>
<p><b>Use:</b> Outputs are labelled synthetic; the academic models are not intended
for impersonation or commercial deployment.</p>

<h2>Source and reproducibility</h2>
<p>Requirements: official Problem-Statement.pdf, pages 1 and 4. The executed
notebook (<code>notebooks/</code>), modular app/core source, pinned Pipfile/requirements,
model downloader, sample CSV, benchmark CSV/JSON, figures, audio, and run
instructions are included. GPU execution is preferred; CPU execution requires
no CUDA and works unchanged on Rocky Linux 9.5, and neither path needs FFmpeg.</p>

<h2>Sample output screenshots</h2>
<div class="shots">
  <figure><img src="data:image/png;base64,{shot_app}"><figcaption>A. Streamlit generated state</figcaption></figure>
  <figure><img src="data:image/png;base64,{shot_wave}"><figcaption>B. Waveform + log-mel</figcaption></figure>
  <figure><img src="data:image/png;base64,{shot_perf}"><figcaption>C. Duration/response/RTF</figcaption></figure>
</div>

</div>
</div>

<div class="foot">
Group 16 | Hamza Aziz | 2024AC05133 | 2024ac05133@wilp.bits-pilani.ac.in &mdash;
<span class="warn">Genuine BITS Virtual Lab screenshot evidence is still pending and must be added after real use.</span>
</div>

</body>
</html>
"""


def main() -> None:
    """Write the report HTML next to this script."""

    out_path = REPO / "Group16_Project_Report_One_Page.html"
    out_path.write_text(build_html(), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
