import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import time
from Operator import BaseOperator
from utils import extract_module_code
import json
import re

class Reviewer(BaseOperator):
    def __init__(self,model: str,temperature: float=0.5, assignment: str=None, topk:int=3, **kwargs):
        super().__init__(model, temperature, topk, **kwargs)
        self.llm = self.get_llm(model, temperature)
        self._cost = {"input_token": 0, "output_tokens":0,"time": 0}
        self.assignment = assignment # Type of the reviewer, can be "RIOT", "IFTTT", "Python" so on. Default is None. If None, the reviewer will use LLM to determine the type.
        if self.assignment not in ["RIOT", "IFTTT", "Python", None]:
            raise ValueError("Invalid type. Type must be one of 'RIOT', 'IFTTT', 'Python' or None.")
    def _run(self, query:str=None):
        """
        Run the Chain of Thought reasoning process on the input or query.
        """
        if query is None:
            raise ValueError("No input provided. Please provide input during initialization or in the run method.")
        if self.assignment is None:
            # Use LLM to determine the type of the reviewer
            # DPO
            self.assignment = "RIOT"

        if  self.assignment == "Python":
            # 规范输入，从raw_code中提取代码，暂时忽略安全问题
            code_list = extract_module_code(query)
            if len(code_list) == 0:
                raise ValueError("No code found in the input query. Please provide a valid code snippet.")
            elif  len(code_list) > 1:
                raise ValueError("Multiple code blocks found in the input query. Please provide only one code snippet.")
            code = code_list[0]
            result = Executor.python_executor(code)
            system_prompt = f"""
            You are an expert Python code reviewer with deep knowledge of Python programming best practices. Your task is to analyze the following code and its execution results.

            In your review, please focus on:
            1. Correctness - Does the code function as intended based on the execution results?
            2. Efficiency - Are there any performance optimizations that could be made?
            3. Readability - Is the code clear and well-structured following PEP 8 guidelines?
            4. Error handling - Does the code properly handle potential exceptions?
            5. Logic - Are there any logical flaws or edge cases not being addressed?

            Provide constructive feedback and explain both strengths and weaknesses of the implementation. 
            When suggesting improvements, be specific and include code examples where appropriate. 
            """
            human_prompt = f"""
            I have a task where I asked you to generate code based on my requirements. You provided a code solution, which I then executed and obtained some results. Now, I would like you to:
            1. Review the code you generated based on the execution results
            2. Identify any issues or inefficiencies in the code
            3. Explain what worked well and what didn't work as expected
            4. Provide a corrected and improved version of the code that addresses any problems found
            Here is the original task and code:
            {query}

            Here is the execution result:
            {result}
            Please provide your review in a concise manner. In the end, please also provide a corrected version of the code.
            """
            messages = [SystemMessage(system_prompt), HumanMessage(human_prompt+"You must wrap the final answer with ```")]
            response = self.llm.invoke(messages)
            self._update_cost(messages,response.content if isinstance(response, BaseMessage) else response)
        elif self.assignment == "RIOT":
            # 规范输入，从raw_code中提取代码，暂时忽略安全问题
            code_list = extract_module_code(query)
            if len(code_list) == 0:
                raise ValueError("No code found in the input query. Please provide a valid code snippet.")
            elif  len(code_list) > 1:
                raise ValueError("Multiple code blocks found in the input query. Please provide only one code snippet.")
            code = code_list[0]
            result = Executor.riot_executor(code)
            system_prompt = """"
            You are an expert RIOT-OS code reviewer with deep knowledge of embedded systems and the RIOT real-time operating system. Your task is to review the following RIOT code and provide professional, technical feedback.

            As a RIOT specialist, you should analyze:
            1. Correctness - Does the code follow RIOT-OS conventions and will it run correctly on embedded devices?
            2. Efficiency - Is the code optimized for resource-constrained environments?
            3. Style - Does it follow RIOT coding standards and best practices?
            4. Safety - Are there potential issues with memory management, interrupts, or other critical embedded systems concerns?
            5. Completeness - Does the implementation satisfy all requirements?

            Your review should include:
            - Identification of bugs or potential issues
            - Analysis of code structure and architecture
            - Suggestions for optimization or improvement
            - Explanation of any RIOT-specific concerns
            - A corrected and improved version of the code if necessary

            Be thorough but practical, focusing on issues that matter most in embedded systems development.
            """
            human_prompt = f"""
            I have a task where I asked you to generate RIOT-OS code based on my requirements. You provided a code solution, which I then executed and obtained some results. Now, I would like you to:
            1. Review the RIOT-OS code you generated based on the execution results
            2. Identify any issues or inefficiencies in the code
            3. Explain what worked well and what didn't work as expected
            4. Suggest improvements or optimizations to the code
            Here is the original task and code:
            {query}
            Here is the execution result:
            {result}
            Please provide your review in a concise manner. In the end, please also provide a corrected version of the code.
            """
            
            if self.topk > 0:
                response = self.retrieval_run(human_prompt, self.topk)
                human_prompt += f"Here is the reference code: " + str(response)
            
            messages = [SystemMessage(system_prompt), HumanMessage(human_prompt+"You must wrap the final answer with ```")]
            response = self.llm.invoke(messages)
            self._update_cost(messages,response.content if isinstance(response, BaseMessage) else response)
        elif self.assignment == "IFTTT":
            result = Executor.ifttt_executor(query)
            system_prompt = """
            You are an expert IFTTT applet reviewer with deep knowledge of IoT integrations and automation workflows. 
            Your task is to analyze the following IFTTT applet configuration and provide professional feedback.

            As an IFTTT specialist, you should analyze:
            1. Correctness - Is the applet configuration valid and will it work as intended?
            2. Efficiency - Is the automation optimized for its purpose?
            3. Logic - Does the trigger-action relationship make sense for the intended use case?
            4. Safety - Are there any potential issues with the triggers or actions?
            5. Completeness - Does the implementation satisfy all requirements?

            Your review should include:
            - Identification of any configuration issues
            - Analysis of the trigger-action relationship
            - Suggestions for improvement or optimization
            - Explanation of any IFTTT-specific concerns
            - A corrected and improved version of the configuration if necessary
            """

            human_prompt = f"""
            I have a task where I asked you to generate an IFTTT applet configuration based on my requirements. 
            You provided a solution, which I then executed and obtained some results. Now, I would like you to:

            1. Review the IFTTT configuration based on the execution results
            2. Identify any issues or inefficiencies in the configuration
            3. Explain what worked well and what didn't work as expected
            4. Suggest improvements or optimizations

            Here is the task:
            {query}
            Here is the execution result:
            {result}
            your output should be a json format like this:
            {{
                "Thought": "Explanation of how the TAP was generated.",
                "Action_type": "Either 'AskUser' or 'Finish'.",
                "Say_to_user": "Natural language response to the user.",
                "TAP": {{
                    "trigger": "...",
                    "condition": "...",
                    "action": "..."
                }}
            }}
            """
            messages = [SystemMessage(system_prompt), HumanMessage(human_prompt+"You must wrap the final answer with ```")]
            response = self.llm.invoke(messages)
            self._update_cost(messages,response.content if isinstance(response, BaseMessage) else response)
            
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
        from config import RIOT_ROOT
        workplace = os.path.join(RIOT_ROOT,'examples','LLM_Gen')
        filepath = os.path.join(RIOT_ROOT,'examples','LLM_Gen','main.c')
        # Write the code to the file
        try:
            with open(filepath,'w') as f:
                f.write(code)
        except Exception as e:
            return f"Error writing code to file: {str(e)}"
        import subprocess

        try:
            process = subprocess.Popen(
                ["make"],
                cwd=workplace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            stdout, stderr = process.communicate(timeout=120)
            # Return the execution result 
            if stderr:
                return f"Execution Error:\n{stderr}"
            elif stdout:
                return f"Execution Output:\n{stdout}"
            else:
                return "Code executed successfully with no output."
        except subprocess.TimeoutExpired:
            return "Execution Error: Code execution timed out after 120 seconds."


    @staticmethod
    def ifttt_executor(code:str):
        required_keys = ["Thought", "Say_to_user", "Action_type"]
        missing_keys = []
        for key in required_keys:
            if key not in code:
                missing_keys.append(key)
        if missing_keys:
            return "Error: Missing required keys: {}".format(", ".join(missing_keys))
        else:
            return "IFTTT JSON is valid and contains all required fields (Thought, Say_to_user, Action_type). Please review the content to determine if it satisfies the specific task requirements."

if __name__ == "__main__":
    
    # node = Reviewer(model="gpt-4o", temperature=0.5)
    # res =  node.run("Given a list of integers, find the maximum product of any two distinct integers in the list.")
    # code = """#include <stdio.h>\nint main()\n{\n    printf("s");\n}"""
    # res = Executor.riot_executor(code)
    # print(res)
    
    json_pattern = re.compile(r'({[\s\S]*})')
    test_json = json_pattern.search("""
    {
        "test": {"test": "test"},
    }
    """)
    code = test_json.group(1)
    try:
        json.loads(code)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    pass