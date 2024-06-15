"""
AudioRecorder.py

This file is responsible for recording audio from a microphone and saving it as a file. It provides the necessary functionality to capture audio input, process it, and store it for further use, such as transcription or analysis.

Functionality:
- Initializes the audio recording settings, such as sample rate and chunk size.
- Provides methods to start and stop recording audio.
- Saves the recorded audio to a specified file format.

Interaction with Other Files:
- **AIWordsAssistantApp.py**: The AIWordsAssistantApp class may use the AudioRecorder to capture audio input from the user. The recorded audio is then transcribed and used to generate word predictions.

Classes and Methods:
- **AudioRecorder**: This class encapsulates all the functionality required to record audio.
  - `__init__(self, filename, sample_rate=44100, chunk_size=1024)`: Initializes the audio recording settings, including the filename, sample rate, and chunk size.
  - `start_recording(self)`: Starts the audio recording process.
  - `stop_recording(self)`: Stops the audio recording process and saves the recorded audio to the specified file.
  - `save_audio(self)`: Saves the recorded audio data to a file in WAV format.

### Troubleshooting

If you are unable to get a recording from a microphone, consider the following steps:

1. **Check Microphone Connection**: Ensure that your microphone is properly connected to your computer. If you are using an external microphone, make sure it is securely plugged in.

2. **Microphone Permissions**: Verify that your operating system has granted the necessary permissions for the Python executable to access the microphone. This may involve adjusting privacy settings on your computer.

3. **Try Another Microphone**: If the current microphone is not working, try using a different microphone. Some microphones may not be compatible or may not be recognized by the Python executable. Switching to another microphone that can be activated by the Python executable might resolve the issue.

4. **Update Drivers**: Ensure that your microphone drivers are up to date. Outdated drivers can sometimes cause issues with audio recording.

5. **Test with Other Software**: Test the microphone with other audio recording software to determine if the issue is specific to the AudioRecorder.py script or a more general problem with the microphone.

6. **Check Python Libraries**: Ensure that all necessary Python libraries for audio recording (e.g., sounddevice) are properly installed and up to date.

By following these troubleshooting steps, you can identify and resolve common issues that may prevent successful audio recording.   

© Matthew J. Hergott
"""


import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
from datetime import datetime
from typing import Callable, List
import logging
from pathlib import Path
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioRecorder:
    def __init__(self, session_id: str, callback: Callable[[str, str], None]):
        self.session_id = session_id
        self.callback = callback
        self.recording = False
        self.stream = None
        self.fname = None
        self.path = None
        self.dtype=np.int16
        self.fs = 44100  # Sample rate
        self.buffer: List[np.ndarray] = []
        self.lock = threading.Lock()
        self.thread = None

    def _get_filename(self) -> str:
        now = datetime.now()
        return f"{self.session_id}_{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}.wav"

    def _save_buffer_to_file(self, buffer: List[np.ndarray], fname: str) -> None:
        with self.lock:
            self.fname = fname
            self.path = Path.cwd() / f"recordings/{self.fname}"
            logging.debug(f"Saving buffer to file {self.path}")
           
            # Copy buffer to a new buffer (deep copy)
            new_buffer = copy.deepcopy(buffer)     
            buffer = []     

            # Save the new buffer to the file
            wav.write(self.path, self.fs, np.concatenate(new_buffer).astype(np.int16))
            logging.debug(f"Finished saving file {self.fname}")

    def _recording_thread(self) -> None:
        while self.recording:
            self.fname = self._get_filename()
            self.buffer = []
            start_time = time.time()
            logging.debug(f"Started recording to file {self.fname}")

            while time.time() - start_time < 30 and self.recording:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            if self.buffer and self.recording:
                self._save_buffer_to_file(self.buffer, self.fname)
                
                if self.recording:
                    self.callback(self.session_id, self.fname)

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logging.warning(f"Audio callback status: {status}")
        with self.lock:
            self.buffer.append(indata[:, 0].copy())

    def _list_recording_devices(self) -> None:
        logging.info(f"default [input, output] device: {sd.default.device}")
        qd = sd.query_devices(kind='input')
        for i, device in enumerate(qd):
            for k, v in qd.items():
                logging.info(f"input device # {i}: {device}, {k}: {v}")

    def start_recording(self) -> None:
        if not self.recording:
            self._list_recording_devices()
            self.recording = True
            self.stream = sd.InputStream(callback=self._audio_callback, 
                                         channels=1, 
                                         samplerate=self.fs, 
                                         dtype=self.dtype)
            self.stream.start()
            self.thread = threading.Thread(target=self._recording_thread)
            self.thread.start()
            logging.info("Recording started")

    def stop_recording(self) -> None:
        if self.recording:
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
            if self.thread:
                self.thread.join()
            # with self.lock:
            #     if self.buffer:
            #         self._save_buffer_to_file(self.buffer, self.fname)
            logging.info("Recording stopped")