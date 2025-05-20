import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


class CoT():
    def __init__(self,model: str,temperature: float=0.5):
        self.llm = get_llm(model, temperature)
        self.system_prompt = f"""
You are a programming assistant that solves problems step by step.        
When solving programming problems:
1. First, understand what the problem is asking for
2. Consider the inputs, expected outputs, and constraints
3. Break the problem down into smaller components
4. Write the outline of your approach
5. Briefly explain key parts of your approach\n
"""
    def _run(self, query=None):
        """
        Run the Chain of Thought reasoning process on the input or query.
        """
        input_text = query
        if input_text is None:
            raise ValueError("No input provided. Please provide input during initialization or in the run method.")
        
        user_query = f"Here is the problem: {input_text}. Please provide a step-by-step reasoning process to solve the problem."
        
        messages = [SystemMessage(self.system_prompt), HumanMessage(user_query)]
        response = self.llm.invoke(messages)
        
        return response
    def _validate(self, query=None):
        cost = {"token": 0, "time": 0}
        #TODO: Implement the validation logic
        # For now, we will just return a dummy cost
        return cost
    def run(self, query=None):
        """
        Return the response in text format.
        """
        response = self._run(query)
        response = response.content
        return response      


if __name__ == "__main__":
    node = CoT(model="gpt-4o", temperature=0.5)
    res =  node.run("Given a list of integers, find the maximum product of any two distinct integers in the list.")
    print(res)
