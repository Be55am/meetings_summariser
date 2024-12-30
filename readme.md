Meeting Minutes Generator
======================

This is a Python script that uses the Whisper model to transcribe audio files and extract key information, such as key points and action items. The script then structures the transcript into meeting minutes and saves them to a JSON file.

Features
--------
* Transcription: Uses the Whisper model to transcribe audio files
* Key Points Extraction: Extracts important sentences from the transcript based on predefined keywords
* Action Items Extraction: Extracts action items from the transcript based on predefined keywords
* Meeting Minutes Generation: Structures the transcript into meeting minutes, including a summary, key points, and action items
* JSON Output: Saves the generated minutes to a JSON file

Usage
-----

To use this script, you need to have the Whisper model installed. You can install it using the following command:

pip install transformers

Then, simply run the script with your audio file path as an argument:

python meeting_minutes_generator.py /path/to/audio/file.mp3

This will generate a JSON file containing the meeting minutes.

Example Output
--------------

Here is an example of the output generated by this script:

```json
{
"metadata": {
"date": "2023-02-22 14:30",
"generated_by": "Whisper-based Meeting Minutes Generator",
"transcript_length": 1234
},
"transcript": "This is a sample transcript...",
"sections": {
"summary": "Meeting transcript generated and analyzed.",
"key_points": [
"The meeting was held on February 22, 2023.",
"The main topic of discussion was the company's financial performance."
],
"action_items": [
"John will follow up with Sarah to discuss the project timeline.",
"Jane will provide an update on the marketing strategy by the end of the week."
]
}
}
```

Limitations
------------

This script has some limitations: it only extracts key points and action items based on predefined keywords, which may not be accurate for all transcripts; it does not perform any sentiment analysis or entity recognition, which could be useful in extracting more valuable information from the transcript.
Future Development
------------------

Some potential future developments for this script include: improving the key points and action items extraction algorithms to make them more accurate; adding support for multiple languages or models; integrating sentiment analysis and entity recognition to extract more valuable information from the transcript; allowing users to customize the output format, such as generating a PDF file instead of a JSON file.