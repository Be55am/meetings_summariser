import json
import logging
import platform
from datetime import datetime

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import MODEL_CONFIG, OUTPUT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeetingMinutesGenerator:
    def __init__(self):
        """Initialize the meeting minutes generator with Whisper model"""
        try:
            # Check if MPS (Apple Silicon GPU) is available
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using NVIDIA GPU (CUDA)")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU - No GPU available")

            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"System platform: {platform.platform()}")

            # Load model and processor
            self.model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_CONFIG["model_id"],
                torch_dtype=torch.float32  # Ensure float32 for MPS compatibility
            ).to(self.device)
            self.processor = WhisperProcessor.from_pretrained(MODEL_CONFIG["model_id"])

            # Maximum length for audio chunks in seconds
            self.chunk_length = 30
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def load_audio(self, audio_path: str):
        """Load audio file"""
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=audio, sr=sr)
            logger.info(f"Audio duration: {duration:.2f} seconds")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise

    def split_audio(self, audio, sr):
        """Split audio into chunks"""
        chunk_length_samples = int(self.chunk_length * sr)
        chunks = []

        for i in range(0, len(audio), chunk_length_samples):
            chunk = audio[i:i + chunk_length_samples]
            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue
            chunks.append(chunk)

        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks

    def transcribe_audio(self, audio_path: str):
        """Transcribe audio file using Whisper model"""
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)

            # Split audio into chunks
            chunks = self.split_audio(audio, sr)

            # Process each chunk
            full_transcription = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

                # Process audio chunk
                input_features = self.processor(
                    chunk,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features.to(self.device)

                # Generate transcription with longer max length
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,  # Increased max length
                    num_beams=5,  # Beam search for better results
                    length_penalty=0.6
                )

                # Decode transcription
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )

                full_transcription.append(transcription[0].strip())

            # Join all transcriptions
            final_transcription = " ".join(full_transcription)
            logger.info(f"Completed transcription, length: {len(final_transcription)} characters")

            return final_transcription

        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

    def extract_key_points(self, transcript: str):
        """Extract key points from transcript"""
        # This is a simple implementation - you might want to use a more sophisticated approach
        sentences = transcript.split('.')
        key_points = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences that might be important (contain key words)
            important_keywords = ['need to', 'should', 'important', 'key', 'main', 'critical', 'decided', 'agreed']
            if any(keyword in sentence.lower() for keyword in important_keywords) and len(sentence) > 20:
                key_points.append(sentence)

        return key_points[:10]  # Limit to top 10 key points

    def extract_action_items(self, transcript: str):
        """Extract action items from transcript"""
        sentences = transcript.split('.')
        action_items = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences that might be action items
            action_keywords = ['will', 'need to', 'should', 'going to', 'must', 'have to']
            if any(keyword in sentence.lower() for keyword in action_keywords) and len(sentence) > 20:
                action_items.append(sentence)

        return action_items[:5]  # Limit to top 5 action items

    def structure_minutes(self, transcript: str):
        """Structure the transcript into meeting minutes"""
        try:
            # Extract key information
            key_points = self.extract_key_points(transcript)
            action_items = self.extract_action_items(transcript)

            # Create structured minutes
            minutes = {
                "metadata": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "generated_by": "Whisper-based Meeting Minutes Generator",
                    "transcript_length": len(transcript)
                },
                "transcript": transcript,
                "sections": {
                    "summary": "Meeting transcript generated and analyzed.",
                    "key_points": key_points,
                    "action_items": action_items,
                    "next_steps": []  # This could be enhanced with more sophisticated analysis
                }
            }

            return minutes
        except Exception as e:
            logger.error(f"Error structuring minutes: {str(e)}")
            raise

    def save_minutes(self, minutes: dict, output_filename: str):
        """Save the generated minutes to a JSON file"""
        try:
            output_path = OUTPUT_CONFIG["minutes_dir"] / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(minutes, f, indent=2, ensure_ascii=False)
            logger.info(f"Minutes saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving minutes: {str(e)}")
            raise


def main():
    try:
        # Initialize generator
        generator = MeetingMinutesGenerator()

        # Example usage (replace with your audio file path)
        audio_path = "../resources/denver_extract.mp3"
        output_filename = f"minutes_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        # Generate transcript
        logger.info("Starting transcription...")
        transcript = generator.transcribe_audio(audio_path)

        # Structure minutes
        logger.info("Structuring minutes...")
        minutes = generator.structure_minutes(transcript)

        # Save minutes
        output_path = generator.save_minutes(minutes, output_filename)
        logger.info(f"Meeting minutes generated successfully: {output_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
