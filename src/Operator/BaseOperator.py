import time
from langchain.schema import BaseMessage
from utils import extract_module_code
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing import Union
from config import *
import tiktoken

class BaseOperator:
    """
    Base class for all operators. 
    """

    def __init__(self, model: str, temperature: float = 0.5, ):
        self.model = model
        self.temperature = temperature
        self._cost = {"input_token": 0, "output_tokens": 0, "time": 0}
    
    def _run(self, query=None):
        """
        Run the operator on the input or query.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    def _update_cost(self, input_text, response_text):
        """
        Update the cost based on locally calculated token count using tiktoken.
        
        Args:
            input_text (str): The input text sent to the model.
            response_text (str): The output text received from the model.
        """
        encoding = tiktoken.encoding_for_model(self.model)
        input_tokens = len(encoding.encode(str(input_text)))
        output_tokens = len(encoding.encode(str(response_text)))
        
        self._cost["input_token"] += input_tokens
        self._cost["output_tokens"] += output_tokens
    def reset_cost(self):
        """
        Reset the cost to zero.
        """
        self._cost = {"input_token": 0, "output_tokens":0,"time": 0}
        return
    def get_cost(self):
        # return the cost
        return self._cost
    def run(self, query=None):
        """
        Return the response in text format.
        """
        start_time = time.time()
        response = self._run(query)
        end_time = time.time()
        self._cost["time"] += end_time - start_time
        response = response.content
        response_format = extract_module_code(response)
        return response_format  
    def get_llm(self, model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
        if model not in CANDIATE_MODEL:
            raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
        else:
            llm = ChatOpenAI(model=model, temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))

        return llm