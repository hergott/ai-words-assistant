"""
AIWordsAssistantApp.py

This file contains the main application logic for the AI Words Assistant. 
It integrates various components such as audio recording, speech-to-text transcription, 
and word prediction to provide a seamless user experience.

Functionality:
- Initializes the application and its components.
- Captures audio input from the user using the AudioRecorder.
- Transcribes the recorded audio to text using a speech-to-text engine.
- Generates word predictions based on the transcribed text.
- Provides a user interface for interacting with the application.

Interaction with Other Files:
- **AudioRecorder.py**: The AIWordsAssistantApp class uses the AudioRecorder to capture 
audio input from the user. The recorded audio is then transcribed and used to generate 
word predictions.

Classes and Methods:
- **AIWordsAssistantApp**: This class encapsulates the main application logic.

### Troubleshooting

If you encounter issues with the AI Words Assistant application, consider the following steps:

1. **Check Dependencies**: Ensure that all necessary dependencies are installed and up to date. 
This may include libraries for audio recording, speech-to-text transcription, and word prediction.

2. **Verify Audio Input**: Ensure that your microphone is properly connected and configured. 
Refer to the troubleshooting steps in AudioRecorder.py for more details.

3. **Check Transcription Service**: If the transcription is not working, verify that the 
speech-to-text engine is properly configured and accessible. This may involve checking API keys, 
network connectivity, and service availability.

By following these troubleshooting steps, you can identify and resolve common issues that 
may arise when using the AI Words Assistant application.

© Matthew J. Hergott
"""

import customtkinter as ctk
import tkinter as tk
import asyncio
import uuid
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from threading import Thread
from PIL import Image
import os
import logging
import re

from React import React
from AudioRecorder import AudioRecorder

from strings import words, description, transcription_error_msg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)

class AIWordsAssistantApp:
    def __init__(self):
        self.previous_images = [None] * 24
        self.recording = False
        self.session_id = ''
        self.last_width = 1000
        self.last_height = 700
        self.app = None
        self.start_button = None
        self.grid_frame = None
        self.word_list_changed = False
        self.words = words
        self.loop = asyncio.new_event_loop()
        self.parsing_audio = False
        self.audio_recorder = None  
        self.exiting = False
        self.current_words = [
            'hospital', 'fire', 'officer', 'emergency', 'doctor', 'nurse', 'help', 'heart', 'breath', 'medicine', 
            'child', 'dog', 'cat', 'food', 'money', 'bank', 'computer', 'phone', 'car', 'word',
            'key', 'newspaper', 'people', 'time'
        ]
        self.react = React() 

        self.setup_directories()
        self.check_images()
        self.initialize_app()
        
    def check_images(self):
        images_found = True
        for word in self.words:
            img_path = self.images_dir / f"{word}.png"
            if not img_path.exists():
                images_found = False
                logging.error(f"Image file not found: {img_path}")
        if images_found:
            logging.info('Images found for all words.')

    def setup_directories(self):
        self.recordings_dir = Path("recordings")
        self.conversations_dir = Path("conversations")
        self.conversation_words_dir = Path("conversation_words")

        self.recordings_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        self.conversation_words_dir.mkdir(exist_ok=True)
        
        if not os.path.exists("images"):
            raise Exception("The 'images' directory does not exist.")
        
        self.images_dir = Path('images')

    def initialize_app(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        self.OpenAI_client = OpenAI(api_key=self.OPENAI_API_KEY)

        self.setup_main_window()
        self.create_widgets()
        self.bind_events()
        self.create_image_grid()
        self.start_event_loop()

    def setup_main_window(self):
        self.app = ctk.CTk()
        self.app.title("AI Words Assistant")
        self.app.geometry("1000x700")
        self.app.protocol("WM_DELETE_WINDOW", lambda: self.on_closing())
        self.app.update()
        self.last_width = self.app.winfo_width()
        self.last_height = self.app.winfo_height()
        self.aspect_ratio = self.last_width / self.last_height

    def create_widgets(self):
        ctk.CTkLabel(self.app, text="AI Words Assistant", font=("Arial", 20)).pack(pady=10)

        self.start_button = ctk.CTkButton(
            self.app, text="Start", command=self.toggle_recording, fg_color="green", hover_color="darkgreen"
        )
        self.start_button.pack(pady=10)
        
        self.words_custom_font = ctk.CTkFont(family='Arial', size=16, weight='normal')
        self.word_label_text = tk.StringVar(value="(words without images will appear here)")
        self.words_label = tk.Label(
            self.app,
            textvariable=self.word_label_text,
            wraplength=600,
            font=self.words_custom_font,
            bg="#D9E8D8"
        )
        self.words_label.pack(pady=10)      

        self.grid_frame = ctk.CTkFrame(self.app)
        self.grid_frame.pack(fill="both", expand=True, padx=5, pady=5)      

        self.description_custom_font = ctk.CTkFont(family='Arial', size=16, weight='normal')
        self.description_label = tk.Label( 
            self.app,
            text=description,
            font=self.description_custom_font,
            wraplength=600
        )
        self.description_label.pack(pady=10, padx=10, fill=ctk.X)

    def bind_events(self):
        self.app.bind("<Configure>", lambda event: self.on_resize(event))

    def start_event_loop(self):
        Thread(target=self.run_async_event_loop, daemon=True).start()

    def run_async_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        uuid_full = str(uuid.uuid4())
        self.session_id = uuid_full[:6]
        self.update_button("Stop", "red", "darkred")
        
        # Create an AudioRecorder instance with callback
        self.audio_recorder = AudioRecorder(self.session_id, self.audio_recorder_callback)
        
        # Start the recording in a new thread
        Thread(target=self.audio_recorder.start_recording, daemon=True).start()

    def stop_recording(self, exiting=False):
        if self.audio_recorder:
            self.audio_recorder.stop_recording() 
        self.delete_history()
        self.recording = False
        if not exiting:
            self.update_button("Start", "green", "darkgreen")          

    def update_button(self, text, color, hover_color):
        self.start_button.configure(text=text, fg_color=color, hover_color=hover_color)

    def on_closing(self):
        self.exiting = True
        self.stop_recording(exiting=True)
        if self.react is not None:
            self.react.agent_executor.max_iterations = 0
            self.react.agent_executor.max_execution_time = 0
            
        try:
            self.app.destroy()
        except tk.TclError as e:
            pass
        
        try:
            self.loop.stop()
        except Exception as e:
            pass

    def on_resize(self, event):
        if event.widget == self.app and (self.last_width != event.width or self.last_height != event.height):
            self.create_image_grid()
            self.last_width = event.width
            self.last_height = event.height
            self.aspect_ratio = self.last_width / self.last_height          

    def create_image_grid(self):
        self.clear_grid()
        self.app.update()
        width, height = self.grid_frame.winfo_width(), self.grid_frame.winfo_height()
        columns, rows = self.calculate_grid_dimensions(width, height)
        self.calculate_image_and_font_size(width, height, columns, rows)
        
        for i, word in enumerate(self.current_words):
            self.add_image_to_grid(i,
                                   word,
                                #    self.image_size,
                                #    self.image_labels_font_size,
                                   columns) 

        self.description_label.config(wraplength=int(width*0.9))     
        self.words_label.config(wraplength=int(width*0.9)) 
        
        self.app.update()            

        self.grid_frame.tk_images = self.previous_images
        self.word_list_changed = False    

    def clear_grid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

    def calculate_grid_dimensions(self, width, height):
        aspect_ratio = width / height * 1.16339869 * 1.5
        if aspect_ratio > 6:
            return 12, 2
        if aspect_ratio > 2.67:
            return 8, 3
        if aspect_ratio > 1.5:
            return 6, 4
        if aspect_ratio > 0.67:
            return 4, 6
        if aspect_ratio > 0.375:
            return 3, 8
        return 2, 12

    def calculate_image_and_font_size(self, width, height, columns, rows):
        xadj = 0.74 + (width - 988) * 0.0001 if width < 988 else 0.74 + (width - 988) * 0.00001
        effective_width = round(width * xadj)
        self.image_size = effective_width // columns
        self.description_font_size = max(8, self.image_size // 6)
        self.words_font_size = max(9, self.image_size // 5)
        self.image_labels_font_size = max(8, self.image_size // 6)
        self.set_words_size()
        return
    
    def set_words_size(self):
        self.words_custom_font.configure(size=self.words_font_size)
        self.words_label.configure(wraplength=int(self.grid_frame.winfo_width()*0.9)) 
        self.description_custom_font.configure(size=self.description_font_size)
        
    def set_words_text(self, text):
        self.word_label_text.set(text)

    def add_image_to_grid(self, index, word, columns):
        image_path = Path.cwd() / f"images/{word}.png"
        if not image_path.exists():
            logging.warning(f"Image not found: {image_path}")
            return
        
        img = ctk.CTkImage(Image.open(image_path), 
                           size=(self.image_size, self.image_size))
        self.previous_images[index] = img

        img_label = ctk.CTkLabel(
            self.grid_frame,
            image=img,
            text=word,
            compound="top",
            font=("Arial", self.image_labels_font_size)
        )
        img_label.grid(row=index // columns, column=index % columns, padx=5, pady=5)
        img_label.bind("<Button-1>", lambda event, img_index=index: self.on_image_click(img_index))

    def on_image_click(self, index):
        logging.info(f"Image {index + 1} clicked")

    def transcribe(self, fname):
        transcription_error = False
        
        file_path = self.recordings_dir / fname
        if not file_path.exists():
            transcription_error = True
            logging.error(f"File {file_path} does not exist.")
            return None, transcription_error

        try:
            with file_path.open("rb") as audio_file:
                transcription = self.OpenAI_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )     
        except Exception as e:
              transcription_error = True
              logging.error(f'Error transcribing audio: {e}')
              logging.error(transcription_error_msg)
              return None, transcription_error
                      
        text = transcription.text
        
        if len(text)<1:
            transcription_error = True
            logging.info(transcription_error_msg)
            return None, transcription_error
        
        logging.info(f"Transcription for {fname}: {text}")

        return text, transcription_error

    def delete_history(self):
        for directory in [self.recordings_dir,
                          self.conversations_dir,
                          self.conversation_words_dir]:
            for file_path in directory.glob(f"*{self.session_id}*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        logging.info(f"Deleted {file_path}")
                    except PermissionError as e:
                        logging.error(f"Failed to delete {file_path}: {e}")


    def update_conversation(self, text):
        conversation_file = self.conversations_dir / f"{self.session_id}.txt"
        old_text = conversation_file.read_text() if conversation_file.exists() else ""
        new_text = (old_text + text)[-4000:]
        
        with open(conversation_file, 'w') as f:
            f.write(new_text)        
        
        # Split the string on commas, whitespace, and any punctuation characters
        conversation_words = re.split("[,\s\.\?!]", new_text)

        # Remove any whitespace characters
        conversation_words = [word.strip() for word in conversation_words]

        # Find the unique string values
        conversation_words = set(conversation_words)

        # Save the unique string values to the file
        conversation_words_file = os.path.join('conversation_words',
                                               f'{self.session_id}.txt')
        with open(conversation_words_file, 'w') as f:
            f.write(','.join(conversation_words))        

        logging.info(f"Updated conversation for session {self.session_id}")
        
        return new_text
        
    def audio_recorder_callback(self, session_id, fname):
        if self.exiting:
            return
        
        if self.parsing_audio:
            return
        
        self.parsing_audio = True
        self.session_id = session_id
        
        text, transcription_error = self.transcribe(fname)
        
        if transcription_error or text is None:
            self.parsing_audio = False
            return
        
        try:
            conversation_text = self.update_conversation(text)
        except Exception as e:
            logging.error(f'Could not update conversation files: {e}')
            self.parsing_audio = False
            return
        
        try:            
            current_words_new, word_candidates_ex_images = self.react.run_agent_for_app(session_id, 
                                                                                        self.current_words, 
                                                                                        conversation_text)
        except Exception as e:
            logging.error(f'React agent failed: {e}')
            self.parsing_audio = False
            return   
        
        if current_words_new is None or word_candidates_ex_images is None:
            self.parsing_audio = False
            return                  
        
        self.word_list_changed = False
        
        for i in range(24):
            if self.current_words[i] != current_words_new[i]:
                self.word_list_changed = True
            self.current_words[i] = current_words_new[i]
        
        try:            
            if self.word_list_changed:
                self.create_image_grid()
                if len (word_candidates_ex_images) < 16:
                    self.set_words_text(', '.join(word_candidates_ex_images))
                else:
                    self.set_words_text(', '.join(word_candidates_ex_images[:16]))
        except Exception as e:
            logging.error(f'Could not update image grid: {e}')
            self.parsing_audio = False
            return                 
 
        self.parsing_audio = False             

if __name__ == "__main__":
    os.chdir('C:/Users/matt_/OneDrive/Documents/Python-projects/NVIDIA-competition/app/')
    try:
        app_instance = AIWordsAssistantApp()
        app_instance.app.mainloop()
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
