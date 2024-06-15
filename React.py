"""
This module implements a ReAct (Reasoning, Action, and Observation) agent using the LangChain Python library. 
The agent is designed to predict important words from conversations and search engine results, and it can be 
integrated into applications to enhance conversational AI capabilities.

A ReAct Agent:
--------------
A ReAct agent is a type of AI agent that follows a specific framework for reasoning, action, and observation. 
The agent first reasons about the input it receives, then takes an action based on that reasoning, and finally
observes the results of the action to inform future decisions. This iterative process allows the agent to
improve its performance over time and handle complex tasks more effectively.

LangChain Python Library:
-------------------------
LangChain is a Python library designed to facilitate the development of language model applications. It provides
tools and abstractions for creating, managing, and deploying language models, making it easier to integrate
advanced NLP capabilities into various applications. LangChain supports multiple language models and offers
features like prompt templates, agent executors, and tool integrations.

Functionality of the Code:
--------------------------
The code defines a ReAct agent that uses the LangChain library to predict important words from conversations
and search engine results. The agent is built using the `ChatNVIDIA` model from the LangChain NVIDIA AI endpoints.
The code includes tools for word prediction and search result analysis, and it integrates these tools into the
ReAct agent framework.

Classes and Functions:
----------------------

1. `WordsPredictLLM(query: str) -> str`:
    - A tool that predicts 50 important words likely to be used in a conversation.
    - Uses the `ChatNVIDIA` model to generate predictions based on the input query.

2. `SearchResultsWordsLLM(query: str) -> str`:
    - A tool that predicts 50 important words related to the results of a search engine query.
    - Uses the `ChatNVIDIA` model to generate predictions based on the search engine results.

3. `React` Class:
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

Usage:
------
To use the ReAct agent, create an instance of the `React` class and call the `run_agent` or `run_agent_for_app`
methods with the appropriate input. The `demo` method can be used to see a demonstration of the agent's capabilities.

Example:
--------
```python
if __name__ == "__main__":
    react = React()
    react.demo(demo_num=1)
```

© Matthew J. Hergott
"""

from dotenv import load_dotenv
import os
import random
import logging

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from TavilyCustom.tool import TavilyAnswer, TavilySearchResults

import strings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool
def WordsPredictLLM(query: str) -> str:
    """Predicts 50 important words that are likely to be used in a conversation.""" 
    
    llm_predict = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1",
                             temperature=0)
    
    llm_query_format = """
    You are an expert at the English language, but you have no knowledge of recent news. 
    You are given an input text of a conversation between two or more people. 
    Predict 50 important words that are likely to be used in this conversation. 
    Give the results as a list of words separated by commas.

    Conversation description: {query}

    Answer: 
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=llm_query_format
    )

    llm_chain = prompt | llm_predict

    result = llm_chain.invoke(query)
    print(result.content)    
    
    return result.content  

@tool
def SearchResultsWordsLLM(query: str) -> str:
    """Finds important words related to results of search engine query.""" 
    
    llm_search_results_words = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1",
                                          temperature=0)
    
    llm_query_format = """
    You are an expert at the English language. 
    You are given the output of a search engine query. 
    Predict 50 important words that are likely to be used in a 
    conversation among people talking about these search engine results.
    Give the results as a list of words separated by commas.

    Search engine results: {query}

    Answer: 
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=llm_query_format
    )

    llm_chain = prompt | llm_search_results_words

    result = llm_chain.invoke(query)
    print(result.content)    
    
    return result.content 

class React:
    def __init__(self) -> None:
        self.words = strings.words
        self.react_template = strings.react_template
        print(self.react_template)
        
        self.agent_llm_model = "mistralai/mixtral-8x7b-instruct-v0.1"
        
        self.load_model()
        self.create_tools()
        self.create_agent()
        
    def load_model(self):
        load_dotenv()
        
        # https://python.langchain.com/v0.1/docs/integrations/chat/nvidia_ai_endpoints/
        self.llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", temperature=0)            
    
    def create_tools(self):
        self.TavilyTool = TavilySearchResults(max_results=3)
        self.tools = [WordsPredictLLM, self.TavilyTool, SearchResultsWordsLLM]
        print(self.tools)
        
    def get_available_models(self):
        return ChatNVIDIA.get_available_models()
    
    def create_agent(self):
        prompt = PromptTemplate(
            template=self.react_template,
            input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools']
        )

        # https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
        self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        # https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, 
                                            verbose=True, handle_parsing_errors=True, 
                                            max_iterations=5, max_execution_time=15)    
        
    def run_agent(self, input):
        agent_error = False
        
        try:
            output = self.agent_executor.invoke({"input": input})
        except Exception as e:
            agent_error = True
            logging.error(f'Error running react agent: {e}') 
            return None, agent_error
        
        if output is None or output['output'] is None or len(output['output'])<1:
            agent_error = True
            logging.error(f'Error running react agent: output is empty.') 
            return None, agent_error            

        output = output['output'].strip().lower()
        
        logging.info(f'Agent output type: {type(output)}.')
        logging.info(f'Agent output: {output}.')

        replace_list = ["Final Answer", "[", "]", ".", ";", ":", "{", "}", "!", "?", "(", ")", "-", "_"]

        for i in replace_list:
            output = output.replace(i, '')

        react_words = output.split(',')
        
        if len(react_words)<3:
            agent_error = True
            logging.error(f'Error running react agent: only returned {len(react_words)} words.') 
            return react_words, agent_error        
        
        for i, react_string in enumerate(react_words):
            react_string_trim = react_string.strip()
            temp = ''.join(char for char in react_string_trim if char.isalpha() or char == ' ')
            react_words[i] = temp.strip()

        # # remove results that have multiple words
        # react_words = [w for w in react_words if ' ' not in w]
        
        react_words = list(set(react_words))
        logging.info(f'React words: {react_words}') 
        
        return react_words, agent_error  
    
    def parse_words_for_app(self, session_id, current_words, react_words):
        # get words previously used in conversation
        filename = os.path.join('conversation_words', f'{session_id}.txt')

        with open(filename, 'r') as f:
            conv_words_str = f.read()
        
        conv_words = conv_words_str.split(',')
        conv_words = [word.strip() for word in conv_words]  
        
        # eliminate from word candidates words previously used in conversation
        filtered_react_words = [word for word in react_words if word not in conv_words]  
        
        # # eliminate from word candidates words currently used as images
        # word_candidates = [word for word in filtered_react_words if word not in current_words]
        
        # find word candidates that have image associated with them
        word_candidates_images = [word for word in filtered_react_words if word in self.words and word not in current_words]
        
        # duplicate list of words currently used as images
        current_words_new = [word for word in current_words]
        
        # replace words used as images, depending on how many results
        if len(word_candidates_images) < 24:
            # Replace a random sample of current_words with all word_candidates_images
            random_indices = random.sample(range(len(current_words)), len(word_candidates_images))
            for i, c in enumerate(random_indices):
                current_words_new[c] = word_candidates_images[i]
        elif len(word_candidates_images) == 24:
            # Replace all current_words with all word_candidates_images
            current_words_new = [word for word in word_candidates_images]
        else:
            # Replace all current_words with a random sample of word_candidates_images
            random_indices = random.sample(range(len(word_candidates_images)), 24)
            current_words_new[:] = [word_candidates_images[i] for i in random_indices]   
        
        word_candidates_ex_images = [word for word in filtered_react_words if word not in current_words_new]
            
        return current_words_new, word_candidates_ex_images
    
    def run_agent_for_app(self, session_id, current_words, conversation_text):
        react_words, agent_error = self.run_agent(conversation_text)
        
        if agent_error:
            return None, None
        
        try:
            current_words_new, word_candidates_ex_images = self.parse_words_for_app(session_id, current_words, react_words)        
        except Exception as e:
            logging.error(f'Error parsing agent output: {e}')
    
        return current_words_new, word_candidates_ex_images
    
    def demo(self, demo_num=0):
        if demo_num==0:
            input = "What happened in the election?\nThe results just came in. I am shocked."
        else:
            input = "I am in pain.\nWere you in the flood? I saw it on the news."
            
        _, _ = self.run_agent(input=input)

# Run
if __name__ == "__main__":
    react = React()   
    react.demo(demo_num=1)            
        
        
    
        