import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import os

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 5  # Record for 5 seconds
        self.channels = 1
        
        # Download and load Vosk model
        if not os.path.exists("model"):
            print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            print("Recommended model: vosk-model-small-en-us-0.15")
            raise RuntimeError("Model not found")
            
        self.model = Model("model")
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        
    def listen(self):
        print("Recording... Speak now!")
        try:
            # Record audio
            recording = sd.rec(
                int(self.sample_rate * self.duration),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16
            )
            sd.wait()  # Wait until recording is finished
            
            # Convert to format expected by Vosk
            self.recognizer.AcceptWaveform(recording.tobytes())
            result = json.loads(self.recognizer.FinalResult())
            
            if 'text' in result:
                return result['text']
            return None
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None 