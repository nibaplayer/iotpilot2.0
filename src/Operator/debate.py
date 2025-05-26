import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from Operator import BaseOperator
from utils import extract_module_code

class DebateOperator(BaseOperator):
    """
    Debate Operator that implements a multi-agent debate system where different agents with
    different roles debate to find better solutions for tasks.

    Parameters:
        model (str): Name of the language model to use.
        temperature (float): Temperature parameter for the language model.
        num_agents (int): Number of agents to work on subtasks.
    """

    def __init__(self, model: str, temperature: float = 0.5, num_agents: int = 2, **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.num_agents = num_agents
        self.llm = self.get_llm(model, temperature=temperature)
        
        # Instructions for debate process
        self.debate_initial_instruction = "Please think step by step and then solve the task. Use ``` to wrap code."
        self.debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Use ``` to wrap code."
        self.final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer. You should generate the complete code. Use ``` to wrap code."
        # Define default agent roles
        self.default_roles = ['Network Expert', 'Sensor Expert']
    def _run(self, query=None):
        """
        Run the debate operator: initial reasoning -> debate rounds -> final decision.
        """
        if query is None:
            raise ValueError("No input provided. Please provide a query/task.")

        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [self.get_llm(self.model, temperature=0.8) for _ in range(self.num_agents)]
        agent_roles = self.default_roles[:self.num_agents]  # Use default roles based on number of agents
        
        max_round = 2  # Maximum number of debate rounds
        all_thinking = [[] for _ in range(max_round)]
        all_answer = [[] for _ in range(max_round)]

        # print(f"[INFO] Starting debate process with {self.num_agents} agents.")
        
        # Perform debate rounds
        for r in range(max_round):
            # print(f"[INFO] Starting debate round {r + 1}/{max_round}")
            
            for i in range(len(debate_agents)):
                if r == 0:
                    # print(f"[DEBUG] Agent '{agent_roles[i]}' starting initial reasoning.")
                    messages = [
                        SystemMessage("You are a helpful assistant and " + agent_roles[i] + ". " + self.debate_initial_instruction),
                        HumanMessage(query)
                    ]
                    response = debate_agents[i].invoke(messages)
                    self._update_cost(messages, response.content)
                    thinking = response.content
                    answer = response.content
                    # print(f"[DEBUG] Agent '{agent_roles[i]}' completed initial reasoning.")
                else:
                    # print(f"[DEBUG] Agent '{agent_roles[i]}' updating solution based on peers' feedback.")
                    input_context = [query] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                    context_str = "\n".join(input_context)
                    
                    messages = [
                        SystemMessage("You are a helpful assistant and " + agent_roles[i] + ". " + self.debate_instruction),
                        HumanMessage(context_str)
                    ]
                    response = debate_agents[i].invoke(messages)
                    self._update_cost(messages, response.content)
                    thinking = response.content
                    answer = response.content
                    print(f"[DEBUG] Agent '{agent_roles[i]}' completed updated solution.")
                    
                all_thinking[r].append(thinking)
                all_answer[r].append(answer)
        
        # print("[INFO] All debate rounds completed. Preparing final decision.")
        
        # Make the final decision based on all debate results and solutions
        final_input = [query] + all_thinking[max_round-1] + all_answer[max_round-1]
        final_input_str = "\n".join(final_input)
        
        messages = [
            SystemMessage("You are a helpful assistant. " + self.final_decision_instruction),
            HumanMessage(final_input_str)
        ]
        final_response = self.llm.invoke(messages)
        self._update_cost(messages, response.content)

        cost = self.get_cost()
        
        return final_response
    
    
if __name__ == "__main__":
    debate_node = DebateOperator(model="gpt-4o", temperature=0.5)
    res = debate_node.run("give me a MQTT-based RIOT application.")
    print(res)
    
    # Extract code block from response
    code_block = extract_module_code(res)  
    
    # Save to output.c
    filename = "output/output.c"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(code_block[0])
    print(f"Saved generated code to {filename}")
    
    print(debate_node.get_cost())