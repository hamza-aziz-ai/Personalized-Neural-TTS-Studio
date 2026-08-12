# Piper voice models

This relocated project already includes both official Piper voice/config pairs:

- en_US-hfc_female-medium.onnx and its .onnx.json configuration
- en_US-hfc_male-medium.onnx and its .onnx.json configuration

The two ONNX files total approximately 126 MB. The lean portal ZIP omits
them and provides download_models.py; the offline-model ZIP includes all four
files. The notebook and application search this local models folder first.

The Hi-Fi Captain model cards identify the voice data license as
CC BY-NC-SA 4.0. This project uses the models for non-commercial academic work.
