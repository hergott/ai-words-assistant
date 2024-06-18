# AI Words Assistant Application

## Overview

The AI Words Assistant is a Python-based graphical user interface (GUI) application designed to enhance conversational experiences by recording conversations, transcribing voice to text, predicting likely words using artificial intelligence, and selecting images based on these predictions. This application integrates various components such as audio recording, speech-to-text transcription, and word prediction to provide a seamless and interactive user experience.

## Instructions

Note: these instructions install the required Python libraries directly into a virtual environment. On some systems, it's necessary to `activate` the virtual environment to install libraries: <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>

1. Git clone or download app (131MB .ZIP file) from: <https://github.com/hergott/ai-words-assistant>

2. If downloaded .ZIP file, unzip app to location of your choice. The app folder is referred to here as `/app`

2. In the `/app/.env` file, enter your API keys for NVIDIA NIM, OpenAI, and Tavily.

3. Create new Python virtual environment: `python -m venv /path/to/new/virtual/environment`

4. Move to virtual environment Scripts folder: `cd /path/to/new/virtual/environment/Scripts`

5. (Optional) Upgrade the pip installer: `python -m pip install --upgrade pip`

6. Install Python libraries into the new virtual environment: `pip install -r /app/requirements.txt`

7. Navigate to the app directory: `cd /app`

8. Run the main file: `/path/to/new/virtual/environment/Scripts/python.exe AIWordsAssistantApp.py`

## Sample output

#### Intro screen

<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/app_entry.png" width=60%>


#### Screen and terminal output after listening to video about pickleball (a sport like tennis)

<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/app_exit.png" width=60%>
<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/gimp_pickleball_1.png" width=60%>
<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/gimp_pickleball_2.png" width=60%>
<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/gimp_pickleball_3.png" width=60%>
<img src="https://github.com/hergott/ai-words-assistant/blob/main/sample_output/gimp_pickleball_full.png" width=60%>

## How This Application Can Help with Word Recall

The AI Words Assistant application can significantly aid individuals in recalling words during conversations in several ways:       

1. **Real-Time Word Prediction**: By predicting important words that are likely to occur in a conversation, the application can provide users with a list of suggested words, helping them recall terms they might otherwise forget.

2. **Contextual Cues**: The application selects images based on predicted words, offering visual cues that can jog the user's memory and assist in recalling specific words or phrases.

3. **Enhanced Vocabulary**: By exposing users to predicted words and their associated contexts, the application can help expand their vocabulary and improve their ability to recall words in future conversations.

4. **Support for Language Learners**: For individuals learning a new language, the application can provide valuable support by predicting and displaying words that are relevant to the conversation, aiding in language acquisition and recall.

5. **Memory Aid for Cognitive Impairments**: Individuals with cognitive impairments or memory issues can benefit from the application's word prediction and visual cues, which can serve as memory aids during conversations.

6. **Preparation for Important Conversations**: Users can use the application to prepare for important conversations by inputting key topics and receiving predicted words and images that can help them recall important points during the actual conversation.        

7. **Interactive Learning Tool**: The application can be used as an interactive learning tool, where users engage in conversations with the AI and receive real-time feedback on word usage and recall, enhancing their conversational skills.

By leveraging these features, the AI Words Assistant application can provide substantial support to individuals looking to improve their word recall and conversational abilities.

## Features

- **Audio Recording**: Captures audio input from the user using a microphone.
- **Speech-to-Text Transcription**: Converts recorded audio into text using a speech-to-text engine.
- **Word Prediction**: Uses AI to predict important words that are likely to occur in the conversation.
- **Image Selection**: Selects images based on the predicted words to enhance the conversational context.
- **User Interface**: Provides an intuitive GUI for interacting with the application.

## Components

### 1. AIWordsAssistantApp.py

This file contains the main application logic for the AI Words Assistant. It integrates various components to provide a cohesive user experience.

#### Functionality

- Initializes the application and its components.
- Captures audio input from the user using the `AudioRecorder`.
- Transcribes the recorded audio to text using a speech-to-text engine.
- Generates word predictions based on the transcribed text.
- Provides a user interface for interacting with the application.

#### Interaction with Other Files

- **AudioRecorder.py**: The `AIWordsAssistantApp` class uses the `AudioRecorder` to capture audio input from the user. The recorded audio is then transcribed and used to generate word predictions.

#### Classes and Methods

- **AIWordsAssistantApp**: This class encapsulates the main application logic.

### 2. AudioRecorder.py

This file is responsible for recording audio from a microphone and saving it as a file. It provides the necessary functionality to capture audio input, process it, and store it for further use, such as transcription or analysis.

#### Functionality

- Initializes the audio recording settings, such as sample rate and chunk size.
- Provides methods to start and stop recording audio.
- Saves the recorded audio to a specified file format.

#### Interaction with Other Files

- **AIWordsAssistantApp.py**: The `AIWordsAssistantApp` class may use the `AudioRecorder` to capture audio input from the user. The recorded audio is then transcribed and used to generate word predictions.

#### Classes and Methods

- **AudioRecorder**: This class encapsulates all the functionality required to record audio.
  - `__init__(self, filename, sample_rate=44100, chunk_size=1024)`: Initializes the audio recording settings, including the filename, sample rate, and chunk size.
  - `start_recording(self)`: Starts the audio recording process.
  - `stop_recording(self)`: Stops the audio recording process and saves the recorded audio to the specified file.
  - `save_audio(self)`: Saves the recorded audio data to a file in WAV format.

### 3. ReAct Agent

This module implements a ReAct (Reasoning, Action, and Observation) agent using the LangChain Python library. The agent is designed to predict important words from conversations and search engine results, and it can be integrated into applications to enhance conversational AI capabilities.

#### Functionality

- Predicts important words from conversations and search engine results using the `ChatNVIDIA` model from the LangChain NVIDIA AI endpoints.
- Integrates tools for word prediction and search result analysis into the ReAct agent framework.

#### Classes and Functions

1. **WordsPredictLLM(query: str) -> str**:
    - A tool that predicts 50 important words likely to be used in a conversation.
    - Uses the `ChatNVIDIA` model to generate predictions based on the input query.

2. **SearchResultsWordsLLM(query: str) -> str**:
    - A tool that predicts 50 important words related to the results of a search engine query.
    - Uses the `ChatNVIDIA` model to generate predictions based on the search engine results.

3. **React** Class:
    - Initializes the ReAct agent, loads the language model, creates tools, and sets up the agent.
    - Methods:
        - `__init__(self)`: Initializes the ReAct agent and sets up the necessary components.
        - `load_model(self)`: Loads the language model using environment variables.
        - `create_tools(self)`: Creates the tools for word prediction and search result analysis.
        - `get_available_models(self)`: Returns the available models from the `ChatNVIDIA` endpoint.
        - `create_agent(self)`: Creates the ReAct agent using the specified tools and prompt template.
        - `run_agent(self, input)`: Runs the agent with the given input and returns the predicted words.
        - `parse_words_for_app(self, session_id, current_words, react_words)`: Parses the predicted words for use in an application.
        - `run_agent_for_app(self, session_id, current_words, conversation_text)`: Runs the agent and parses the output for an application.
        - `demo(self, demo_num=0)`: Demonstrates the agent's functionality with a sample input.

## Usage

To use the ReAct agent, create an instance of the `React` class and call the `run_agent` or `run_agent_for_app` methods with the appropriate input. The `demo` method can be used to see a demonstration of the agent's capabilities.

### Example

```python
if __name__ == "__main__":
    react = React()
    react.demo(demo_num=1)
```

## Troubleshooting

### AI Words Assistant Application

1. **Check Dependencies**: Ensure that all necessary dependencies are installed and up to date. This may include libraries for audio recording, speech-to-text transcription, and word prediction.
2. **Verify Audio Input**: Ensure that your microphone is properly connected and configured. Refer to the troubleshooting steps in `AudioRecorder.py` for more details.
3. **Check Transcription Service**: If the transcription is not working, verify that the speech-to-text engine is properly configured and accessible. This may involve checking API keys, network connectivity, and service availability.

### Audio Recording

1. **Check Microphone Connection**: Ensure that your microphone is properly connected to your computer. If you are using an external microphone, make sure it is securely plugged in.
2. **Microphone Permissions**: Verify that your operating system has granted the necessary permissions for the Python executable to access the microphone. This may involve adjusting privacy settings on your computer.
3. **Try Another Microphone**: If the current microphone is not working, try using a different microphone. Some microphones may not be compatible or may not be recognized by the Python executable.
4. **Update Drivers**: Ensure that your microphone drivers are up to date. Outdated drivers can sometimes cause issues with audio recording.
5. **Test with Other Software**: Test the microphone with other audio recording software to determine if the issue is specific to the `AudioRecorder.py` script or a more general problem with the microphone.
6. **Check Python Libraries**: Ensure that all necessary Python libraries for audio recording (e.g., sounddevice) are properly installed and up to date.

By following these troubleshooting steps, you can identify and resolve common issues that may arise when using the AI Words Assistant application.

© Matthew J. Hergott
