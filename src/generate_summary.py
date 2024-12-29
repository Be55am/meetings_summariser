import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_transcript(json_file_path):
    """Load the transcript from the JSON file."""
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return data.get('transcript', '')
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def generate_summary(transcript, max_length=4096):
    """Generate a summary using Llama model."""
    # Set device to MPS (Metal Performance Shaders) for Mac M chips
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize tokenizer and model
    model_name = "meta-llama/Llama-3.1-8B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto"
        )
        model.to(device)

        # Prepare the prompt
        prompt = f"Please provide a concise summary of the following transcript:\n\n{transcript}\n\nSummary:"

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate summary
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and return the summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split("Summary:")[1].strip()

    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def main():
    # File paths
    json_file = "/Users/bessam/IdeaProjects/ai_generated/llms_lab/llms_testing/output/minutes/minutes_20241229_1951.json"

    # Load transcript
    transcript = load_transcript(json_file)
    if not transcript:
        print("Failed to load transcript")
        return

    # Generate summary
    summary = generate_summary(transcript)
    if summary:
        print("\nGenerated Summary:")
        print("-----------------")
        print(summary)
    else:
        print("Failed to generate summary")

if __name__ == "__main__":
    main()