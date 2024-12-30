# Meeting Minutes Generator

A Python-based tool that automatically generates meeting minutes from audio recordings using Whisper for transcription and Llama for summarization.

## Features

- Automatic speech-to-text transcription using OpenAI's Whisper model
- GPU acceleration support (CUDA for NVIDIA, MPS for Apple Silicon)
- Audio chunking for processing long recordings
- Extraction of key points and action items
- Meeting minutes summarization using Meta's Llama model
- Structured output in both JSON and Markdown formats

## Requirements

- Python 3.8+
- PyTorch
- librosa
- transformers
- Meta Llama model access (for summarization)
- NVIDIA GPU with CUDA support (optional)
- Apple Silicon Mac (optional, for MPS acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd meeting-minutes-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up model access:
- Ensure you have access to Meta's Llama model
- Configure your model credentials in the appropriate configuration files

## Project Structure

```
meeting-minutes-generator/
├── src/
│   ├── config.py
│   ├── meeting_minutes_generator.py
│   └── summarizer.py
├── resources/
│   └── audio_files/
├── output/
│   ├── minutes/
│   └── summaries/
└── README.md
```

## Usage

1. Place your audio file in the `resources/audio_files` directory.

2. Run the transcription and initial minutes generation:
```python
python src/meeting_minutes_generator.py
```

3. Generate the summary from the JSON transcript:
```python
python src/summarizer.py
```

The tool will:
- Transcribe the audio file
- Generate structured minutes in JSON format
- Create a summarized version in Markdown format

## Configuration

The project uses two main configuration files:

1. `MODEL_CONFIG`: Contains settings for the Whisper model
2. `OUTPUT_CONFIG`: Defines output directories and file formats

Modify these configurations in `config.py` according to your needs.

## Output Format

### JSON Output
The tool generates a JSON file containing:
- Metadata (date, generator info)
- Full transcript
- Key points
- Action items
- Next steps

### Markdown Summary
The summary includes:
- Meeting date and time
- Attendees and location
- Discussion points
- Key takeaways
- Action items with owners

## GPU Acceleration

The tool automatically detects and uses available GPU acceleration:
- NVIDIA GPUs via CUDA
- Apple Silicon via MPS
- Falls back to CPU if no GPU is available

## Error Handling

The tool includes comprehensive error handling and logging:
- Detailed error messages
- Activity logging
- Exception handling for audio processing and model operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2024 Be55am

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

- OpenAI's Whisper model for transcription
- Meta's Llama model for summarization
- The librosa team for audio processing capabilities