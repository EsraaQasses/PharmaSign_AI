# PharmaSign Speech-to-Text

## What it does

This project transcribes pharmacy audio files into Arabic text using `openai/whisper-large-v3`.

It includes:
- audio and split extraction from zip files
- CSV path fixing
- Arabic text normalization
- simple VAD-based preprocessing
- transcription and evaluation using WER and CER

## Files

- `data.py`: extracts dataset files and fixes CSV paths
- `app.py`: loads the model, preprocesses audio, and transcribes batches
- `evaluate.py`: runs evaluation and saves predictions
- `requirements.txt`: required libraries

## How to run

```bash
pip install -r requirements.txt
python data.py
python evaluate.py
