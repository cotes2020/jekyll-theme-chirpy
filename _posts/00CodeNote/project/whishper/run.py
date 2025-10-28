import whisper

# Load the Whisper model
model = whisper.load_model("small")  # use "medium" or "large" for higher accuracy

# Path to your audio file
audio_path = "./provider/fast_run/inter.m4a"

# Transcribe
result = model.transcribe(audio_path)

# Extract only the dialogue text
text = result["text"].strip()

# Save to file
with open("transcript_cleaned.txt", "w") as f:
    f.write(text)

print("✅ Transcription complete. Saved as transcript_cleaned.txt")
