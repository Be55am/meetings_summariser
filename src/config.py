from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Model configurations
MODEL_CONFIG = {
    "model_id": "openai/whisper-small.en",
    "task": "automatic-speech-recognition",
}

# Output configurations
OUTPUT_CONFIG = {
    "minutes_dir": ROOT_DIR / "output" / "minutes",
    "transcripts_dir": ROOT_DIR / "output" / "transcripts",
}

# Create output directories if they don't exist
for directory in OUTPUT_CONFIG.values():
    directory.mkdir(parents=True, exist_ok=True)
