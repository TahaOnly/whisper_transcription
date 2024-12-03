import os
from faster_whisper import WhisperModel

model_size = "large-v3"

# Specify the input and output directories
input_dir = "./Recordings"  # Directory containing the audio files
output_dir = "./Transcriptions"  # Directory to save the transcriptions

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the Whisper model
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Loop through each file in the recordings folder
for filename in os.listdir(input_dir):
    if filename.endswith(".mp3") or filename.endswith(".wav"):  
        audio_path = os.path.join(input_dir, filename)
        transcription_filename = os.path.splitext(filename)[0] + ".txt"  # Save as same name with .txt extension
        transcription_path = os.path.join(output_dir, transcription_filename)

        print(f"Processing {filename}...")

        # Transcribe the audio file
        segments, info = model.transcribe(audio_path, beam_size=5, language='en')

        print(f"Detected language '{info.language}' with probability {info.language_probability}")

        full_transcription = ""

        # Iterate over each segment and concatenate the text
        for segment in segments:
            full_transcription += segment.text + " "  # Add a space between segments

        # Write the concatenated transcription to a text file in the output folder
        with open(transcription_path, "w", encoding="utf-8") as file:
            file.write(full_transcription.strip())  # Write the text without trailing spaces

        # Print the full concatenated transcription to the console
        print("\nFull Transcription:")
        print(full_transcription.strip())
        print("\n" + "="*50)  # Separator for better readability

print("All files have been processed.")
