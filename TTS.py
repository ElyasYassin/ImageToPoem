import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties to make the voice sound more poetic
engine.setProperty('rate', 130)  # Speed of speech (words per minute)
engine.setProperty('volume', 0.5)  # Volume level (0.0 to 1.0)

# Note: pyttsx3 may not support changing pitch directly. 
# This functionality varies depending on the TTS engine used.

# Poetic text
poetic_text = """
   The light of a candle
   Is transferred to another candle —
   spring twilight
"""

# Save the spoken text to an audio file
output_path = "output.mp3"
engine.save_to_file(poetic_text, output_path)

# Run the speech engine to process the file
engine.runAndWait()

print(f"Speech generated and saved as {output_path}")
