import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def load_transcript(json_file_path):
    """Load the transcript from the JSON file."""
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return data.get('transcript', '')
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def generate_summary(transcript):
    """Generate a summary using Llama model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_name = "meta-llama/Llama-3.1-8B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.to(device)

        prompt = (f"Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways;"
                  f" and action items with owners of the following transcript:\n\n{transcript}\n\nSummary:")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Changed parameters to use max_new_tokens instead of max_length
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Controls length of the generated summary
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split("Summary:")[1].strip()

    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def save_summary_markdown(summary, output_dir, input_filename):
    """Save the summary as a markdown file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        markdown_filename = f"summary_{base_name}_{timestamp}.md"
        markdown_path = os.path.join(output_dir, markdown_filename)

        markdown_content = f"""# Meeting Summary

## Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}

{summary}

---
*This summary was automatically generated using Llama-3.1-8B model.*
"""

        with open(markdown_path, 'w') as file:
            file.write(markdown_content)

        print(f"\nSummary saved to: {markdown_path}")
        return markdown_path

    except Exception as e:
        print(f"Error saving markdown file: {e}")
        return None

def main():
    json_file = "/Users/bessam/IdeaProjects/ai_generated/llms_lab/llms_testing/output/minutes/minutes_20241229_1951.json"
    output_dir = "/Users/bessam/IdeaProjects/ai_generated/llms_lab/llms_testing/output/summaries"

    transcript = load_transcript(json_file)
    if not transcript:
        print("Failed to load transcript")
        return

    summary = generate_summary(transcript)
    if summary:
        print("\nGenerated Summary:")
        print("-----------------")
        print(summary)

        markdown_path = save_summary_markdown(summary, output_dir, json_file)
        if not markdown_path:
            print("Failed to save summary as markdown")
    else:
        print("Failed to generate summary")

if __name__ == "__main__":
    main()