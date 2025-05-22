import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_llm
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import time
from .BaseOperator import BaseOperator

class Reflexion(BaseOperator):
    """
    Reflexion Agent.
    
    This agent iteratively improves its solutions through self-reflection.
    It generates an initial solution, then evaluates and refines it through
    multiple iterations, providing a more accurate and comprehensive final answer.
    
    Parameters:
        model (str): Name of the language model to use.
        temperature (float|None): Temperature parameter for the language model.
        max_iterations (int): Maximum number of reflection iterations (default=3).
    """
    def __init__(self, model: str, temperature: float=0.5, max_iterations: int=3):
        super().__init__(model=model, temperature=temperature)
        self.initial_llm = get_llm(model, temperature=temperature)
        self.reflect_llm = get_llm(model, temperature=0.3)  # Lower temperature for stable reflections
        self.system_initial_prompt = """
You are an expert problem solver who thinks step-by-step.
When solving problems:
1. First, understand what the problem is asking for
2. Consider the inputs, expected outputs, and constraints
3. Break the problem down into smaller components
4. Outline your approach
5. Briefly explain key parts of your solution
"""
        self.system_reflect_prompt = """
You are a code review and improvement expert. Your task is to:
1. Carefully evaluate the previous solution
2. Identify flaws, gaps, or potential improvements
3. Reflect on how to address these issues
4. Provide an improved, more comprehensive solution
5. Ensure the new solution is more accurate, efficient, and robust
"""
        self.system_final_prompt = """
You are a final solution provider. Your task is to:
1. Carefully evaluate the previous solution
2. Identify the key points and concepts
3. Provide a concise, thorough, and easy-to-understand final solution
"""
        self.max_iterations = max_iterations
    
    def _run(self, query=None):
        """
        Run the self-reflection process, improving the solution through multiple iterations.
        
        Args:
            query (str, optional): The problem or task to solve.
            
        Returns:
            BaseMessage: The final solution after multiple iterations of reflection.
            
        Raises:
            ValueError: If no input query is provided.
        """
        input_text = query
        if input_text is None:
            raise ValueError("No input provided. Please provide input during initialization or in the run method.")
        
        # First round: Generate initial solution
        user_query = f"Problem: {input_text}\n\nPlease provide a detailed step-by-step analysis to solve this problem."
        
        messages = [SystemMessage(self.system_initial_prompt), HumanMessage(user_query)]
        current_solution = self.initial_llm.invoke(messages)
        response_id = current_solution.response_metadata['id']
        self._update_cost(current_solution)
        
        # Subsequent rounds: Reflection and improvement
        for i in range(self.max_iterations - 1):
            reflection_prompt = f"""
I've previously attempted to solve this problem:

Problem: {input_text}

My solution was:

{current_solution.content}

Please evaluate this solution:
1. What parts are correct?
2. What flaws or oversights exist?
3. What important concepts or edge cases did I miss?
4. How can this solution be improved?

Provide an improved, more comprehensive solution.
"""
            # manually add the history of the previous response. need to use memory component in langgraph
            new_messages = [SystemMessage(self.system_reflect_prompt), HumanMessage(reflection_prompt)]
            messages.extend([current_solution]+new_messages)
            improved_solution = self.reflect_llm.invoke(messages)
            # response_id = improved_solution.response_metadata['id']
            self._update_cost(improved_solution)
            
            # Set improved solution as current for next iteration
            current_solution = improved_solution
        
        # Final summary, integrating all reflections
        final_prompt = f"""
Problem: {input_text}

After several iterations of thinking and improvement, I need to provide a final, comprehensive solution.
Please synthesize all previous thinking and provide a concise yet thorough final answer, ensuring the solution is correct, efficient, and easy to understand.
"""
        new_messages = [SystemMessage(self.system_final_prompt), HumanMessage(final_prompt)]
        messages.extend([current_solution]+new_messages)
        final_solution = self.reflect_llm.invoke(messages)
        self._update_cost(final_solution)
        
        return final_solution


if __name__ == "__main__":
    node = Reflexion(model="gpt-4o", temperature=0.5, max_iterations=3)
    res = node.run("Given a list of integers, find the maximum product of any two distinct integers in the list.")
    print(res)
    print("Cost information:", node.get_cost())
    