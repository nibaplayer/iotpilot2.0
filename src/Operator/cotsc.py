import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import time
from .BaseOperator import BaseOperator
from utils import extract_module_code

class CoTSC(BaseOperator):
    """
    Chain of Thought with Self Consistency (CoT-SC) agent.
    
    This agent generates multiple reasoning paths using the Chain of Thought approach,
    then synthesizes a final answer by analyzing the consistency across all paths.
    Self-consistency helps to improve accuracy by aggregating multiple reasoning chains.
    
    Parameters:
        model (str): Name of the language model to use.
        temperature (float|None): Temperature parameter for the language model.
        N (int): Number of different reasoning paths to generate (default=5).
    """
    def __init__(self,model: str,temperature: float=0.5, N:int=5):
        super().__init__(model=model, temperature=temperature)
        self.cot_llm = self.get_llm(model, temperature=0.7)
        self.final_llm = self.get_llm(model, temperature=0.1)
        self.system_cot_prompt = f"""
                                You are a programming assistant that solves problems step by step.        
                                When solving programming problems:
                                1. First, understand what the problem is asking for
                                2. Consider the inputs, expected outputs, and constraints
                                3. Break the problem down into smaller components
                                4. Write the outline of your approach
                                5. Briefly explain key parts of your approach\n
                                """
        self.system_final_prompt = f"""
                                Given all the above solutions, reason over them carefully and provide a final answer.
                                """
        self.N = N  # Number of reasoning paths to generate
    def _run(self, query=None):
        """
        Run the Chain of Thought reasoning process with Self Consistency on the input query.
        Generate N different responses with higher temperature and select the most consistent one.
        """
        input_text = query
        if input_text is None:
            raise ValueError("No input provided. Please provide input during initialization or in the run method.")
        
        user_query = f"Here is the problem: {input_text}. Please provide a step-by-step reasoning process to solve the problem."
        
        all_responses = []
        solution_content = ""
        # Generate N different responses with the same prompt but different temperature
        for i in range(self.N):
            # Adjust temperature for diversity in reasoning paths
            temp_llm = self.get_llm(model="gpt-4o", temperature=0.7)
            messages = [SystemMessage(self.system_cot_prompt), HumanMessage(user_query)]
            response = temp_llm.invoke(messages)
            all_responses.append(response)
            self._update_cost(messages, response)
            solution_content += f"SOLUTION {i+1}:\n{response.content}\n"
        # Synthesize a final answer using all collected responses
        synthesis_prompt = f"""
                            I have generated {self.N} different solutions to this problem using chain-of-thought reasoning.
                            Here are all the solutions:

                            {solution_content}

                            Please carefully analyze all these solutions. Identify the most common elements, correct any mistakes, 
                            and synthesize a final, accurate solution to the original problem:

                            {input_text}

                            Provide a clear, step-by-step reasoning process for this final solution.
                            """
        # Use a lower temperature for the final synthesis to ensure stability
        final_llm = self.final_llm
        messages = [SystemMessage(self.system_final_prompt), HumanMessage(synthesis_prompt)]
        final_response = final_llm.invoke(messages)
        self._update_cost(messages, response)
        return final_response
     


if __name__ == "__main__":
    node = CoTSC(model="gpt-4o", temperature=0.5, N=3)
    res = node.run("Given a list of integers, find the maximum product of any two distinct integers in the list.")
    print(node.get_cost())
