import time
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
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
    def _update_cost(self,response: BaseMessage):
        """
        Update the cost based on the response.
        """
        if self.model in ["gpt-4o"]:
            current_cost = response.response_metadata['token_usage']
            self._cost["input_token"] += current_cost['prompt_tokens']
            self._cost["output_tokens"] += current_cost['completion_tokens']
        return
    def reset_cost(self):
        """
        Reset the cost to zero.
        """
        self._cost = {"input_token": 0, "output_tokens":0,"time": 0}
        return
    def get_cost(self):
        #TODO: Implement the validation logic
        # For now, we will just return a dummy cost
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
        return response  