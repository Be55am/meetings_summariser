import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
    user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcript}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer)

    # Remove quantization config and load model normally
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16  # Use float16 instead of 4-bit quantization
    )

    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
    response = tokenizer.decode(outputs[0])
    return response

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
    json_file = "../output/minutes/minutes_20241229_1951.json"
    output_dir = "../output/summaries"

    transcript = load_transcript(json_file)
    if not transcript:
        print("Failed to load transcript")
        return

    summary = generate_summary(transcript)
    if summary:
        print("\nGenerated Summary:")
        print("-----------------")
        print(summary)
        # split by .<|eot_id|><|start_header_id|>assistant<|end_header_id|> and get second part
        content = summary.split(".<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        second_part = content[1].split("<|eot_id|>")[0]

        print(second_part)

        markdown_path = save_summary_markdown(second_part, output_dir, json_file)
        if not markdown_path:
            print("Failed to save summary as markdown")
    else:
        print("Failed to generate summary")

if __name__ == "__main__":
    main()