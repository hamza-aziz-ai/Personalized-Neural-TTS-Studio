"""Core neural text-to-speech and audio-evaluation utilities for Assignment 2.

The module separates reusable speech-processing logic from the Streamlit user
interface. Piper performs neural synthesis, preferring GPU (CUDA) execution
and falling back to CPU automatically; librosa applies an independent pitch
shift and computes diagnostic acoustic features.

Course: Speech Processing (S2-25 AIMLCZG520)
Group: Group 16
Author: Hamza Aziz (2024AC05133)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


def _register_cudnn_dll_directories() -> None:
    """Make pip-installed cuDNN/cuBLAS DLLs discoverable on Windows.

    onnxruntime-gpu does not bundle cuDNN; the ``nvidia-cudnn-cu12`` and
    ``nvidia-cublas-cu12`` wheels install the DLLs under site-packages, but
    Windows only finds them via ``PATH`` or ``os.add_dll_directory`` (cuDNN 9's
    lazy sub-library loader ignores directories added the other way alone).
    A no-op everywhere else, including the BITS Virtual Lab, where these
    wheels are not installed and the CPU fallback is expected to run.
    """

    if sys.platform != "win32":
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in (site_packages / "cudnn" / "bin", site_packages / "cublas" / "bin"):
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_register_cudnn_dll_directories()

import librosa
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import soundfile as sf
from matplotlib.figure import Figure
from piper import PiperVoice, SynthesisConfig


VOICE_CATALOG: dict[str, dict[str, str]] = {
    "Female": {
        "model_id": "en_US-hfc_female-medium",
        "model_file": "en_US-hfc_female-medium.onnx",
        "display_name": "Hi-Fi Captain Female (US English)",
        "license": "CC BY-NC-SA 4.0",
    },
    "Male": {
        "model_id": "en_US-hfc_male-medium",
        "model_file": "en_US-hfc_male-medium.onnx",
        "display_name": "Hi-Fi Captain Male (US English)",
        "license": "CC BY-NC-SA 4.0",
    },
}

DEFAULT_TEXT = (
    "Speech processing enables machines to understand and generate human "
    "communication clearly and efficiently."
)
MAX_TEXT_CHARACTERS = 5_000
MIN_RATE_MULTIPLIER = 0.70
MAX_RATE_MULTIPLIER = 1.40
MIN_PITCH_SEMITONES = -4.0
MAX_PITCH_SEMITONES = 4.0


@dataclass(frozen=True)
class SynthesisResult:
    """Container for generated speech and its measured characteristics.

    Attributes:
        wav_bytes: PCM-16 WAV data suitable for playback or download.
        waveform: Mono floating-point samples in the interval ``[-1, 1]``.
        sample_rate: Number of audio samples per second in hertz.
        voice_label: Human-readable voice category selected by the user.
        text: Sanitized text that was synthesized.
        rate_multiplier: Requested speaking-rate multiplier.
        pitch_semitones: Post-synthesis pitch change in semitones.
        latency_seconds: End-to-end synthesis and processing time.
        metrics: Objective timing and acoustic diagnostic measurements.
    """

    wav_bytes: bytes
    waveform: np.ndarray
    sample_rate: int
    voice_label: str
    text: str
    rate_multiplier: float
    pitch_semitones: float
    latency_seconds: float
    metrics: dict[str, float]


def validate_text(text: str, max_characters: int = MAX_TEXT_CHARACTERS) -> str:
    """Validate and lightly normalize text before neural speech synthesis.

    This step is needed because empty input produces no audio, while very long
    unbounded input can make an interactive CPU application appear unresponsive.
    Printable whitespace is collapsed without removing useful punctuation that
    guides prosody.

    Args:
        text: Raw text entered by the user.
        max_characters: Maximum allowed length after whitespace normalization.

    Returns:
        A stripped, single-spaced string that is safe to synthesize.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If the result is empty or exceeds ``max_characters``.
    """

    if not isinstance(text, str):
        raise TypeError("Text input must be a string.")

    # Replace control whitespace with ordinary spaces while retaining sentence
    # punctuation, which Piper uses to infer pauses and intonation.
    normalized_text = re.sub(r"\s+", " ", text).strip()

    if not normalized_text:
        raise ValueError("Enter at least one word before generating speech.")
    if len(normalized_text) > max_characters:
        raise ValueError(
            f"Text contains {len(normalized_text)} characters; the limit is "
            f"{max_characters}."
        )

    return normalized_text


def resolve_model_path(model_directory: Path | str, voice_label: str) -> Path:
    """Resolve and validate the ONNX file for a selected Piper voice.

    Piper requires both ``model.onnx`` and the adjacent ``model.onnx.json``
    configuration. Checking both files here turns a cryptic runtime failure into
    a precise setup instruction.

    Args:
        model_directory: Directory containing downloaded Piper voice files.
        voice_label: Key from :data:`VOICE_CATALOG`, such as ``"Female"``.

    Returns:
        Absolute or relative :class:`~pathlib.Path` to the requested ONNX model.

    Raises:
        ValueError: If the voice label is not supported.
        FileNotFoundError: If either required model file is missing.
    """

    if voice_label not in VOICE_CATALOG:
        supported_voices = ", ".join(VOICE_CATALOG)
        raise ValueError(
            f"Unknown voice '{voice_label}'. Choose one of: {supported_voices}."
        )

    model_path = Path(model_directory) / VOICE_CATALOG[voice_label]["model_file"]
    config_path = model_path.with_suffix(model_path.suffix + ".json")

    missing_paths = [path for path in (model_path, config_path) if not path.exists()]
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths)
        model_id = VOICE_CATALOG[voice_label]["model_id"]
        raise FileNotFoundError(
            f"Missing {missing_names}. Run `python -m piper.download_voices "
            f"--data-dir models {model_id}` first."
        )

    return model_path


def load_piper_voice(model_path: Path | str, prefer_gpu: bool = True) -> PiperVoice:
    """Load one Piper ONNX voice model for repeated inference.

    Model loading is separated from synthesis because loading a roughly 63 MB
    neural network is much slower than reusing an already loaded model. The
    Streamlit layer caches this function's result.

    GPU is the preferred execution target when an NVIDIA CUDA provider is
    installed and usable. A CUDA session can load successfully yet still fail
    the first time a kernel actually runs (e.g. a missing/mismatched cuDNN
    DLL), so this performs one silent warm-up synthesis to confirm the GPU
    path genuinely executes before trusting it; any failure there falls back
    to the CPU provider, which is always installed and matches the BITS
    Virtual Lab environment.

    Args:
        model_path: Path to a Piper ``.onnx`` model with adjacent JSON config.
        prefer_gpu: Try CUDA execution first when available, falling back to
            CPU on any load or runtime failure.

    Returns:
        A ready-to-use :class:`piper.PiperVoice` instance. Call
        ``voice.session.get_providers()[0]`` to see which execution provider
        is actually active.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: Propagated by Piper if CPU ONNX initialization fails.
    """

    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Piper model not found: {resolved_path}")

    if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            gpu_voice = PiperVoice.load(resolved_path, use_cuda=True)
            next(gpu_voice.synthesize("Warm up."))  # forces a real kernel run
            return gpu_voice
        except Exception as exc:  # pragma: no cover - depends on local CUDA/cuDNN install
            warnings.warn(f"CUDA execution unavailable ({exc}); falling back to CPU.")

    return PiperVoice.load(resolved_path, use_cuda=False)


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono floating-point waveform as a portable PCM-16 WAV byte stream.

    Args:
        waveform: One-dimensional audio samples, normally in ``[-1, 1]``.
        sample_rate: Sampling frequency in hertz.

    Returns:
        Complete WAV file contents as ``bytes``.

    Raises:
        ValueError: If the waveform is empty, non-mono, or sample rate is invalid.
    """

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("A non-empty mono waveform is required.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    # Clipping before PCM conversion prevents integer wrap-around distortion.
    audio = np.clip(audio, -1.0, 1.0)
    wav_buffer = BytesIO()
    sf.write(wav_buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_buffer.getvalue()


def compute_audio_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    latency_seconds: float,
    word_count: int,
) -> dict[str, float]:
    """Compute interpretable timing and signal-quality diagnostics.

    The assignment asks for response time and audio quality. Without a clean
    reference recording, intrusive metrics such as PESQ or STOI would be invalid.
    The function therefore reports reproducible non-reference diagnostics:
    duration, real-time factor, loudness proxies, clipping, silence, spectral
    centroid, and zero-crossing rate.

    Args:
        waveform: One-dimensional mono floating-point audio samples.
        sample_rate: Sampling frequency in hertz.
        latency_seconds: Measured end-to-end generation time.
        word_count: Number of whitespace-delimited words in the input text.

    Returns:
        Dictionary of scalar metrics suitable for a table or Streamlit cards.

    Raises:
        ValueError: If audio is empty or timing inputs are invalid.
    """

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Cannot evaluate an empty or multi-channel waveform.")
    if sample_rate <= 0 or latency_seconds < 0:
        raise ValueError("Sample rate must be positive and latency non-negative.")

    epsilon = np.finfo(np.float32).eps
    duration_seconds = audio.size / float(sample_rate)
    rms_amplitude = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak_amplitude = float(np.max(np.abs(audio)))
    rms_dbfs = 20.0 * math.log10(max(rms_amplitude, epsilon))
    peak_dbfs = 20.0 * math.log10(max(peak_amplitude, epsilon))

    # A sample at or above 0.999 is close enough to full scale to flag possible
    # clipping after floating-point-to-PCM conversion.
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))

    frame_length = 2_048
    hop_length = 512
    frame_rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    silence_threshold = 10.0 ** (-45.0 / 20.0)
    silence_ratio = float(np.mean(frame_rms < silence_threshold))

    spectral_centroid_hz = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    )
    zero_crossing_rate = float(
        np.mean(librosa.feature.zero_crossing_rate(y=audio))
    )

    return {
        "duration_seconds": duration_seconds,
        "latency_seconds": latency_seconds,
        "real_time_factor": latency_seconds / max(duration_seconds, epsilon),
        "estimated_wpm": (word_count / max(duration_seconds, epsilon)) * 60.0,
        "sample_rate_hz": float(sample_rate),
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": peak_dbfs - rms_dbfs,
        "clipping_percent": clipping_ratio * 100.0,
        "silence_percent": silence_ratio * 100.0,
        "spectral_centroid_hz": spectral_centroid_hz,
        "zero_crossing_rate": zero_crossing_rate,
    }


def synthesize_speech(
    voice: PiperVoice,
    text: str,
    voice_label: str,
    rate_multiplier: float = 1.0,
    pitch_semitones: float = 0.0,
    volume: float = 1.0,
    max_characters: int = MAX_TEXT_CHARACTERS,
    on_chunk: Callable[[int], None] | None = None,
) -> SynthesisResult:
    """Generate personalized speech and measure end-to-end performance.

    Piper controls phoneme duration with ``length_scale``. The intuitive rate
    multiplier is therefore mapped as ``length_scale = 1 / rate_multiplier``.
    Piper has no native pitch parameter, so librosa performs a phase-vocoder pitch
    shift after synthesis while preserving duration as closely as possible.

    Piper yields one audio chunk per sentence rather than all at once. Longer
    input therefore produces multiple chunks; ``on_chunk`` is invoked after each
    one is received so a caller (e.g. the Streamlit UI) can render live progress
    instead of blocking silently until the entire text has synthesized.

    Args:
        voice: Previously loaded Piper neural voice model.
        text: User-provided text to synthesize.
        voice_label: Voice category used for metadata and the interface.
        rate_multiplier: Speaking speed in ``[0.70, 1.40]``; 1.0 is normal.
        pitch_semitones: Pitch change in ``[-4, 4]`` semitones; 0 is unchanged.
        volume: Linear output-volume multiplier accepted by Piper.
        max_characters: Maximum accepted input length; forwarded to
            :func:`validate_text` so the UI's configurable limit is enforced.
        on_chunk: Optional callback invoked with the running chunk count
            (starting at 1) as each sentence finishes synthesizing.

    Returns:
        :class:`SynthesisResult` containing audio, settings, latency, and metrics.

    Raises:
        ValueError: If text or any customization parameter is outside its range.
        RuntimeError: If Piper returns no audio chunks.
    """

    normalized_text = validate_text(text, max_characters=max_characters)
    if voice_label not in VOICE_CATALOG:
        raise ValueError(f"Unsupported voice label: {voice_label}")
    if not MIN_RATE_MULTIPLIER <= rate_multiplier <= MAX_RATE_MULTIPLIER:
        raise ValueError(
            f"Rate must be between {MIN_RATE_MULTIPLIER:.2f} and "
            f"{MAX_RATE_MULTIPLIER:.2f}."
        )
    if not MIN_PITCH_SEMITONES <= pitch_semitones <= MAX_PITCH_SEMITONES:
        raise ValueError(
            f"Pitch must be between {MIN_PITCH_SEMITONES:.1f} and "
            f"{MAX_PITCH_SEMITONES:.1f} semitones."
        )
    if not 0.5 <= volume <= 1.5:
        raise ValueError("Volume must be between 0.5 and 1.5.")

    synthesis_config = SynthesisConfig(
        length_scale=1.0 / rate_multiplier,
        normalize_audio=True,
        volume=volume,
    )

    start_time = perf_counter()
    audio_chunks = []
    for chunk in voice.synthesize(normalized_text, syn_config=synthesis_config):
        audio_chunks.append(chunk)
        if on_chunk is not None:
            on_chunk(len(audio_chunks))
    if not audio_chunks:
        raise RuntimeError("Piper returned no audio for the supplied text.")

    sample_rate = int(audio_chunks[0].sample_rate)
    waveform = np.concatenate(
        [chunk.audio_float_array for chunk in audio_chunks]
    ).astype(np.float32)

    if not math.isclose(pitch_semitones, 0.0, abs_tol=1e-9):
        # Semitone conversion follows the equal-tempered scale: a shift of n
        # semitones corresponds to a frequency factor of 2 ** (n / 12).
        waveform = librosa.effects.pitch_shift(
            waveform,
            sr=sample_rate,
            n_steps=pitch_semitones,
            bins_per_octave=12,
        ).astype(np.float32)

    waveform = np.clip(waveform, -1.0, 1.0)
    wav_bytes = waveform_to_wav_bytes(waveform, sample_rate)
    latency_seconds = perf_counter() - start_time
    word_count = len(normalized_text.split())
    metrics = compute_audio_metrics(
        waveform,
        sample_rate,
        latency_seconds,
        word_count,
    )

    return SynthesisResult(
        wav_bytes=wav_bytes,
        waveform=waveform,
        sample_rate=sample_rate,
        voice_label=voice_label,
        text=normalized_text,
        rate_multiplier=float(rate_multiplier),
        pitch_semitones=float(pitch_semitones),
        latency_seconds=latency_seconds,
        metrics=metrics,
    )


def create_audio_diagnostic_figure(
    waveform: np.ndarray,
    sample_rate: int,
    title: str,
) -> Figure:
    """Create waveform and log-mel-spectrogram plots for visual inspection.

    The waveform exposes duration, pauses, and amplitude. The mel spectrogram
    approximates human frequency perception and makes speech formants and temporal
    structure visible.

    Args:
        waveform: One-dimensional mono audio signal.
        sample_rate: Sampling frequency in hertz.
        title: Descriptive label shown above the plots.

    Returns:
        A Matplotlib :class:`~matplotlib.figure.Figure` with two stacked axes.
    """

    audio = np.asarray(waveform, dtype=np.float32)
    time_axis = np.arange(audio.size) / float(sample_rate)
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1_024,
        hop_length=256,
        n_mels=80,
        fmax=sample_rate / 2,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), constrained_layout=True)
    figure.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(time_axis, audio, color="#176B87", linewidth=0.8)
    axes[0].set(
        xlabel="Time (seconds)",
        ylabel="Amplitude",
        title="Waveform",
        xlim=(0, max(time_axis[-1], 0.01)),
    )
    axes[0].grid(alpha=0.20)

    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set(title="Log-mel spectrogram", xlabel="Time (seconds)")
    figure.colorbar(image, ax=axes[1], format="%+2.0f dB", pad=0.01)

    return figure


def save_synthesis_result(
    result: SynthesisResult,
    output_wav_path: Path | str,
) -> tuple[Path, Path]:
    """Save generated WAV audio and a JSON sidecar with settings and metrics.

    Args:
        result: Completed synthesis result to persist.
        output_wav_path: Destination ending in ``.wav``.

    Returns:
        Tuple containing the WAV path and its JSON metadata path.

    Raises:
        ValueError: If the requested audio path does not end in ``.wav``.
    """

    wav_path = Path(output_wav_path)
    if wav_path.suffix.lower() != ".wav":
        raise ValueError("Output path must use the .wav extension.")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(result.wav_bytes)

    metadata_path = wav_path.with_suffix(".json")
    metadata = {
        "voice": result.voice_label,
        "text": result.text,
        "rate_multiplier": result.rate_multiplier,
        "pitch_semitones": result.pitch_semitones,
        "sample_rate": result.sample_rate,
        "metrics": result.metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return wav_path, metadata_path


def append_mos_feedback(
    csv_path: Path | str,
    voice_label: str,
    rate_multiplier: float,
    pitch_semitones: float,
    overall_mos: int,
    naturalness: int,
    clarity: int,
    comments: str = "",
) -> Path:
    """Append one genuine listener evaluation to a MOS CSV file.

    Ratings are never synthesized or pre-filled: a human listener must submit the
    form in the application. This preserves academic integrity while making data
    collection reproducible.

    Args:
        csv_path: Destination feedback file.
        voice_label: Voice evaluated by the listener.
        rate_multiplier: Rate setting used for the evaluated sample.
        pitch_semitones: Pitch setting used for the evaluated sample.
        overall_mos: Overall Mean Opinion Score from 1 (bad) to 5 (excellent).
        naturalness: Perceived naturalness rating from 1 to 5.
        clarity: Perceived clarity rating from 1 to 5.
        comments: Optional qualitative listener feedback.

    Returns:
        Path to the updated CSV file.

    Raises:
        ValueError: If any rating falls outside the 1-to-5 MOS scale.
    """

    ratings = (overall_mos, naturalness, clarity)
    if any(rating not in range(1, 6) for rating in ratings):
        raise ValueError("MOS, naturalness, and clarity must each be integers 1-5.")

    feedback_path = Path(csv_path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = feedback_path.exists()
    fieldnames = [
        "timestamp_utc",
        "voice",
        "rate_multiplier",
        "pitch_semitones",
        "naturalness_1_to_5",
        "clarity_1_to_5",
        "overall_mos_1_to_5",
        "comments",
    ]

    with feedback_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "voice": voice_label,
                "rate_multiplier": f"{rate_multiplier:.2f}",
                "pitch_semitones": f"{pitch_semitones:.1f}",
                "naturalness_1_to_5": naturalness,
                "clarity_1_to_5": clarity,
                "overall_mos_1_to_5": overall_mos,
                "comments": comments.strip(),
            }
        )

    return feedback_path


def summarize_quality(metrics: dict[str, float]) -> list[str]:
    """Translate numeric diagnostics into cautious, readable observations.

    Args:
        metrics: Output from :func:`compute_audio_metrics`.

    Returns:
        List of evidence-based observations. These checks are diagnostic and do
        not claim to replace human intelligibility or naturalness judgments.
    """

    observations: list[str] = []
    if metrics["clipping_percent"] < 0.01:
        observations.append("No material full-scale clipping was detected.")
    else:
        observations.append(
            f"Clipping affected {metrics['clipping_percent']:.3f}% of samples."
        )

    if metrics["real_time_factor"] < 1.0:
        observations.append("Generation was faster than real-time playback.")
    else:
        observations.append("Generation was slower than real-time playback.")

    observations.append(
        "Objective diagnostics require human MOS ratings for a naturalness verdict."
    )
    return observations

