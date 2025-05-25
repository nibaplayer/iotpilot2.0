import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from .BaseOperator import BaseOperator
from utils import extract_module_code

class EnsembleOperator(BaseOperator):
    """
    Ensemble Operator that decomposes a task into multiple subtasks,
    processes each subtask independently using multiple agents,
    and then merges the results to produce a final output.

    Parameters:
        model (str): Name of the language model to use.
        temperature (float): Temperature parameter for the language model.
        num_agents (int): Number of agents to work on subtasks.
    """

    def __init__(self, model: str, temperature: float = 0.5, num_agents: int = 3):
        super().__init__(model=model, temperature=temperature)
        self.num_agents = num_agents
        self.llm = self.get_llm(model, temperature=temperature)
        self.system_decompose_prompt = """
        You are an AI assistant that breaks down complex tasks into smaller, manageable subtasks.
        Given the following task:

        "{task}"

        Please must break it down into {num_agents} distinct and independent subtasks (modules) that can be processed in parallel.
        List them clearly and concisely.

        Please write your modules answer using ```module.
        Note, each moudule contains a function. The total number of modules is limited to {num_agents} or fewer.
        For example:
        ```module
        void *thread1_function(void *arg) {{
            (void)arg;
            while (1) {{
                printf("hello_world thread2\n");
                xtimer_sleep(1);
            }}
            return NULL;
        }}
        ```

        ```module
        void *thread2_function(void *arg) {{
            (void)arg;
            while (1) {{
                printf("hello_world thread1\n");
                xtimer_sleep(1);
            }}
            return NULL;
        }}
        ```

        """

        self.system_merge_prompt = """
        Given the following subtask results:

        {results}

        Synthesize a coherent and comprehensive final answer to the original task:

        "{task}"
        """

    def _run(self, query=None):
        """
        Run the ensemble operator: decompose -> execute -> merge.
        """
        if query is None:
            raise ValueError("No input provided. Please provide a query/task.")

        # Step 1: Decompose task into subtasks
        decomposition_prompt = self.system_decompose_prompt.format(task=query, num_agents=self.num_agents)
        messages = [SystemMessage("You are a helpful assistant."), HumanMessage(decomposition_prompt)]
        decomposition_response = self.llm.invoke(messages)
        self._update_cost(messages, decomposition_response)
                
        subtasks = extract_module_code(decomposition_response.content)
        if len(subtasks) != self.num_agents:
            raise ValueError(f"{len(subtasks)} subtasks generated. Need exactly {self.num_agents}.")

        # Step 2: Execute each subtask independently
        all_subtask_results = []
        for i, subtask in enumerate(subtasks):
            print(f"Processing subtask {i+1}/{self.num_agents}: {subtask}")
            messages = [SystemMessage("You are a helpful assistant.  You need to generate complete code. Use ``` to wrap the code."), HumanMessage(subtask)]
            response = self.llm.invoke(messages)
            self._update_cost(messages, response)
            all_subtask_results.append(f"SUBTASK {i+1}:\n{response.content}")

        # Step 3: Merge results into a final answer
        merge_prompt = self.system_merge_prompt.format(results='\n\n'.join(all_subtask_results), task=query)
        messages = [
            SystemMessage("You are a helpful assistant. You need to generate complete code. Use ``` to wrap the code.\""),
            HumanMessage(merge_prompt)
        ]
        final_response = self.llm.invoke(messages)
        self._update_cost(messages, final_response)

        return final_response


if __name__ == "__main__":
    ensemble_node = EnsembleOperator(model="gpt-4o", temperature=0.5, num_agents=1)
    res = ensemble_node.run("give me a mqtt-based RIOT application.")
    print(res)
    code_block = extract_module_code(res)
    print(code_block)
    print(ensemble_node.get_cost())
    
    filename = "output/output.c"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(code_block[0])
    print(f"Saved generated code to {filename}")