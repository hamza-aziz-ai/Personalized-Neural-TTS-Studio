"""Download the two official Piper voice models required by the project.

Run once from the submission directory:

    python download_models.py

The command downloads approximately 126 MB in total into ``models/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


VOICE_MODEL_IDS = (
    "en_US-hfc_female-medium",
    "en_US-hfc_male-medium",
)


def download_voice_models(model_directory: Path | str = "models") -> Path:
    """Download both Piper ONNX voice/config pairs with the official downloader.

    Args:
        model_directory: Folder in which the four model files will be stored.

    Returns:
        Resolved directory containing the downloaded voice files.

    Raises:
        subprocess.CalledProcessError: If the official downloader exits non-zero.
    """

    destination = Path(model_directory)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "piper.download_voices",
        "--data-dir",
        str(destination),
        *VOICE_MODEL_IDS,
    ]
    subprocess.run(command, check=True)
    return destination.resolve()


if __name__ == "__main__":
    downloaded_directory = download_voice_models()
    print(f"Voice models are ready in: {downloaded_directory}")

