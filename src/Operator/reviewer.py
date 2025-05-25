import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import time
from Operator import BaseOperator
from utils import extract_module_code

class Reviewer(BaseOperator):
    def __init__(self,model: str,temperature: float=0.5, assignment: str=None):
        super().__init__(model, temperature)
        self.llm = self.get_llm(model, temperature)
        self._cost = {"input_token": 0, "output_tokens":0,"time": 0}
        self.assignment = assignment # Type of the reviewer, can be "RIOT", "IFTTT", "Python" so on. Default is None. If None, the reviewer will use LLM to determine the type.
        if self.assignment not in ["RIOT", "IFTTT", "Python", None]:
            raise ValueError("Invalid type. Type must be one of 'RIOT', 'IFTTT', 'Python' or None.")
    def _run(self, query=None):
        """
        Run the Chain of Thought reasoning process on the input or query.
        """
        if query is None:
            raise ValueError("No input provided. Please provide input during initialization or in the run method.")
        if self.assignment is None:
            # Use LLM to determine the type of the reviewer
            # DPO
            self.assignment = "Python"
        if  self.assignment == "Python":
            # 规范输入，从query中提取代码，暂时忽略安全问题
            code_list = extract_module_code(query)
            if len(code_list) == 0:
                raise ValueError("No code found in the input query. Please provide a valid code snippet.")
            elif  len(code_list) > 1:
                raise ValueError("Multiple code blocks found in the input query. Please provide only one code snippet.")
            code = code_list[0]
            result = Executor.python_executor(code)
            system_prompt = f"""
            You are a Python code reviewer. Your task is to review the following code and provide feedback on its correctness, efficiency, and style.
            """
            human_prompt = f"""
            There are a python code snippet and its execution result. Please review the code.
            Here is the code:
            ```python
            {code}
            ```
            Here is the execution result:
            {result}
            Please provide your review in a concise manner. In the end, please also provide a corrected version of the code.
            """
            messages = [SystemMessage(system_prompt), HumanMessage(human_prompt)]
            response = self.llm.invoke(messages)
            self._update_cost(response)

        return response
     
class Executor():
    """
    Executor class for executing code and returning the result or error.
    """
    def __init__(self):
        pass

    @staticmethod
    def python_executor(code:str):
        """
        Execute Python code using a specific Python environment and return the result.
        This simulates executing code in a terminal and captures all output.
        
        Args:
            code (str): Python code to execute.
            
        Returns:
            str: Output of the execution or error message.
        """
        import tempfile
        import subprocess
        import os
        
        # Path to the specific Python interpreter
        python_interpreter = "/home/hao/miniconda3/envs/python_executor/bin/python"
        
        # Check if the interpreter exists
        if not os.path.exists(python_interpreter):
            return f"Error: Python interpreter not found at {python_interpreter}"
        
        try:
            # Create a temporary file to store the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(code)
            
            # Execute the code with the specified Python interpreter and capture output
            # Using shell=True to simulate terminal execution and capture all output
            process = subprocess.Popen(
                f"{python_interpreter} {temp_filename}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
            )
            stdout, stderr = process.communicate(timeout=10)  # 10 seconds timeout
            
            # Remove the temporary file
            os.unlink(temp_filename)
            
            # Return the execution result
            if stderr:
                return f"Execution Error:\n{stderr}"
            elif stdout:
                return f"Execution Output:\n{stdout}"
            else:
                return "Code executed successfully with no output."
                
        except subprocess.TimeoutExpired:
            # Handle timeout
            try:
                os.unlink(temp_filename)
            except:
                pass
            return "Execution Error: Code execution timed out after 10 seconds."
            
        except Exception as e:
            # Handle other exceptions
            try:
                os.unlink(temp_filename)
            except:
                pass
            return f"Execution Error: {str(e)}"
    
    @staticmethod
    def riot_executor(code:str):
        pass

    @staticmethod
    def ifttt_executor(code:str):
        pass







if __name__ == "__main__":
    # node = Reviewer(model="gpt-4o", temperature=0.5)
    # res =  node.run("Given a list of integers, find the maximum product of any two distinct integers in the list.")
    code = f"""print("Hello, World!")"""
    res = Executor.python_executor(code)
    print(res)
