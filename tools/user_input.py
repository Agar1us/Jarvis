from smolagents.tools import Tool
from typing import Any
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import time
import gigaam

asr_model = gigaam.load_model("v2_rnnt")

# TODO try whisper 3 turbo
def listen_and_transcribe(silence_threshold=500, sample_rate=16000):
    """
    Listens to the microphone, detects speech followed by 2 seconds of silence,
    then transcribes the speech using the GigaAM RNNT 2 model.
    
    Args:
        silence_threshold (int): RMS threshold to consider as silence (adjust as needed).
        sample_rate (int): Sampling rate for audio recording (should match model's expected rate).
    
    Returns:
        str: Transcribed text from the audio.
    """
    channels = 1
    dtype = 'int16'
    min_silence_duration = 2  # seconds
    buffer = []
    last_active_time = None
    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype)
    stream.start()
    try:
        while True:
            data, overflowed = stream.read(int(sample_rate * 0.5))  # Read 0.5-second chunks
            if data.size > 0:
                buffer.append(data.copy())
                rms = np.sqrt(np.mean(np.square(data, dtype=np.float32)))
                if rms >= silence_threshold:
                    last_active_time = time.time()
                else:
                    if last_active_time and (time.time() - last_active_time) >= min_silence_duration:
                        break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop()
        stream.close()

    if not buffer:
        return ""
    audio_data = np.concatenate(buffer, axis=0)
    
    if last_active_time:
        total_duration = len(audio_data) / sample_rate
        silence_end = total_duration - (last_active_time - (time.time() - total_duration))
        samples_to_trim = int(silence_end * sample_rate)
        if samples_to_trim > 0:
            audio_data = audio_data[:-samples_to_trim]
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        wavfile.write(tmpfile.name, sample_rate, audio_data)
        text = asr_model.transcribe(tmpfile.name)
    
    return text

USER_INPUT_DESCRIPTION = """
Asks for user's input on a specific question
Returns a string received from the user
Use this tool in cases when you need to request any data

Usage examples:
>>> action_or_answer = user_input("The element in the picture, which is a screenshot of the screen, was not found. Doesn't know what to do to complete the user's task")
>>> print(action_or_answer)
>>> The element really wasn't on the page, then try clicking on the button with the text Search

Important rule: don't abuse this tool!
Before using it, think very carefully whether you really need to use it
"""


class UserInputTool(Tool):
    name = "user_input"
    description = USER_INPUT_DESCRIPTION
    inputs = {
        "question": 
        {
            "type": "string",
            "description": "The question to ask the user"
        }
    }
    output_type = "string"

    def forward(self, question):
        print(question)
        user_input = listen_and_transcribe()
        return user_input
    
# Example usage
if __name__ == "__main__":
    text = listen_and_transcribe()
    print("Transcribed Text:", text)