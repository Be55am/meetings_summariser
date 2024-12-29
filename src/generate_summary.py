from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptSummarizer:
    def __init__(self):
        """Initialize the summarizer with Llama-2 8B model"""
        try:
            # Check for Apple Silicon GPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using NVIDIA GPU (CUDA)")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU - No GPU available")

            # Load model and tokenizer
            model_name = "meta-llama/Llama-3.1-8B"  # You'll need Hugging Face access token
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def generate_summary(self, transcript: str) -> str:
        """Generate a markdown summary of the transcript"""
        try:
            prompt = f"""Below is a transcript from a meeting. Please create a detailed markdown summary that includes:
            - Main topics discussed
            - Key decisions made
            - Action items
            - Notable quotes
            - Participants mentioned

            Transcript:
            {transcript}

            Please format the summary in markdown with appropriate headers and bullet points.
            """

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate summary
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def save_summary(self, summary: str, output_dir: Path):
        """Save the summary as a markdown file"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = output_dir / f"meeting_summary_{timestamp}.md"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            logger.info(f"Summary saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            raise

def main():
    try:
        # Initialize summarizer
        summarizer = TranscriptSummarizer()
        
        # Read the latest transcript from the minutes folder
        minutes_dir = Path("/users/bessam/ideaprojects/ai_generated/llms_lab/llms_testing/output/minutes")
        latest_minutes = max(minutes_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
        
        with open(latest_minutes, 'r', encoding='utf-8') as f:
            minutes_data = json.load(f)
            transcript = minutes_data["transcript"]
        
        # Generate summary
        logger.info("Generating summary...")
        summary = summarizer.generate_summary(transcript)
        
        # Save summary
        output_dir = Path("/users/bessam/ideaprojects/ai_generated/llms_lab/llms_testing/output/summaries")
        summarizer.save_summary(summary, output_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
