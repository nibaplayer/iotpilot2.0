import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import json
import numpy as np
from Operator import cot, cotsc, reflexion, debate, ensemble, reviewer
import random
import networkx as nx
from utils import load_workflow_config, extract_module_code, get_llm, draw_all_individuals_topology
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from multiprocessing import Pool 
import copy
import time
import threading


# Define benchmark tasks
benchmark =[
    "motion_sensor_80",
    # "ControlRoom",
    # "RIOT_DHT11"
]

# Load dataset
data = pd.read_csv("/home/hao/code/iotpilot2.0/datesets/app.csv")

def create_operator_search_space():
    """
    Create the search space for all operators.
    The space includes parameters like model, number of nodes, and iterations.
    """
    search_space = {
        "cot": {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "temperature": [0,1]
        },
        "cotsc": {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "temperature": [0,1],
            "N": [1, 5]  # Different numbers of reasoning paths
        },
        "reflexion": {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "temperature": [0,1],
            "max_iterations": [1, 5]  # Different numbers of iterations
        },
        "debate": {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "temperature": [0,1],
            "num_agents": [1, 2]  # Different numbers of agents
        },
        "ensemble": {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "temperature": [0,1],
            "num_agents": [1, 5]  # Different numbers of subtask agents
        }
    }
    return search_space

def initialize_operators():
    """
    Build a collection of operators by instantiating various operator types.
    """
    operators = {
        "cot": cot.CoT(model="gpt-4o", temperature=0.5),
        "cotsc": cotsc.CoTSC(model="gpt-4o", temperature=0.5, N=3),
        "reflexion": reflexion.Reflexion(model="gpt-4o", temperature=0.5, max_iterations=2),
        "debate": debate.DebateOperator(model="gpt-4o", temperature=0.5, num_agents=2),
        "ensemble": ensemble.EnsembleOperator(model="gpt-4o", temperature=0.5, num_agents=2)
    }
    return operators

def evaluate_operator_performance(operators, benchmark_tasks):
    """
    Evaluate each operator's performance on benchmark tasks.
    """
    results = {}

    for type_name, problem in zip(data['code_name'], data["problem"]):
        if type_name not in benchmark_tasks:
            continue
        print(f"Evaluating operators on task: {type_name}")
        task_result = {type_name: {}}
        for op_name, operator in operators.items():
            try:
                print(f"Running {op_name} on {type_name}...")
                response = operator.run(problem)
                cost_info = operator.get_cost()
                operator.reset_cost() 
                print(f"Cost for {op_name} on {type_name}: {cost_info}")

                task_result[type_name][op_name] = {
                    "cost_info": cost_info,
                    "response": response
                }

            except Exception as e:
                print(f"Error running {op_name} on {type_name}: {str(e)}")
                task_result[type_name][op_name] = {
                    "error": str(e)
                }
            results_dir = "./results"
            json_path = os.path.join(results_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump(task_result, f, indent=2)

    return results


def build_workflow_with_config(workflow_config):
    
    """
    Build a workflow using the given configuration that defines both topology and parameters.
    
    Args:
        workflow_config (dict): A dictionary defining each node's type, parameters, and next nodes.
    
    Returns:
        dict: A dictionary representing the workflow with instances and topology.
    """
    workflow = {}

    G = nx.DiGraph()
    for node_name, node_info in workflow_config.items():
        for next_node in node_info.get("next", []):
            G.add_edge(node_name, next_node)

    try:
        execution_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise ValueError("拓扑结构中存在环，无法排序")

    print("Workflow Execution order:", execution_order)

    for node_name in execution_order:
        if node_name == "end":
            continue

        node_info = workflow_config[node_name]
        op_type = node_info["type"]
        selected_params = node_info["params"]

        if op_type == "cot":
            instance = cot.CoT(**selected_params)
        elif op_type == "cotsc":
            instance = cotsc.CoTSC(**selected_params)
        elif op_type == "reflexion":
            instance = reflexion.Reflexion(**selected_params)
        elif op_type == "debate":
            instance = debate.DebateOperator(**selected_params)
        elif op_type == "ensemble":
            instance = ensemble.EnsembleOperator(**selected_params)
        elif op_type == "reviewer":
            instance = reviewer.Reviewer(**selected_params)
        else:
            raise ValueError(f"未知 operator 类型: {op_type}")

        workflow[node_name] = {
            "type": op_type,
            "instance": instance,
            "params": selected_params,
            "next": node_info.get("next", []),
            "outputs": None
        }

    return workflow

def detect_task_type(problem: str) -> str:
    """
    使用 LLM 判断 problem 的任务类型: IFTTT, RIOT, 或 Python_executer
    
    Args:
        problem (str): 用户提供的自然语言问题描述
        
    Returns:
        str: 任务类型，可选值为 "IFTTT", "RIOT", "Python_executer"
    """
    llm = get_llm(model="gpt-4o", temperature=0.2)
    
    system_prompt = """
    You are a helpful assistant that classifies user problems into one of the following categories:

    - **IFTTT**: Problems involving creating rules/triggers/actions for IoT devices (e.g., "Turn on the light when motion is detected").
    - **RIOT**: Problems requiring formal logic or event-based reasoning over time (e.g., "If no motion for 10 minutes, turn off lights").
    - **Python_executer**: Problems that require writing executable Python code (e.g., "Write a function to calculate Fibonacci numbers").

    Respond ONLY with one of these three words: "IFTTT", "RIOT", or "Python_executer".
    """
    user_prompt = f"""
    Classify the following problem into IFTTT, RIOT, or Python_executer:

    "{problem}"
    """

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_prompt)
    ]
    
    response = llm.invoke(messages)
    task_type = response.content.strip()
    
    valid_types = ["IFTTT", "RIOT", "Python_executer"]
    if task_type in valid_types:
        print(f"Detected task type: {task_type}")
        return task_type
    else:
        print(f"Unknown task type: {task_type}, defaulting to IFTTT")
        return "IFTTT"


def execute_single_run(args):
    """
    Top-level function for multiprocessing.
    Takes a tuple of (i, workflow_config, problem)
    """
    
    i, workflow_config, problem, code_name, method, iter_num = args
    print(f"Executing run {i} for problem: {code_name}")
    
    delay = i * random.uniform(5, 10)
    print(f"Process {i} will start after {delay:.2f} seconds.")
    time.sleep(delay)

    # 子进程中实例化 workflow
    local_workflow = build_workflow_with_config(workflow_config)

    executed = set()
    total_input_token = 0
    total_output_token = 0
    total_time_cost = 0
    node_costs = {}

    def execute_node(node_name, input_data, problem):
        nonlocal total_input_token, total_output_token, total_time_cost
        if node_name in executed or node_name == "end":
            return
        node = local_workflow[node_name]
        executed.add(node_name)

        response = node["instance"].run(input_data)
        print(f"Run {node_name} with params: {node['params']} end")
        
        info = node["instance"].get_cost()
        node["instance"].reset_cost()

        node_input = float(info["input_token"])
        node_output = float(info["output_tokens"])
        node_time = float(info["time"])

        total_input_token += node_input
        total_output_token += node_output
        total_time_cost += node_time

        node_costs[node_name] = {
            "input_token": str(node_input),
            "output_token": str(node_output),
            "time": str(node_time)
        }
        
        node["outputs"] = response

        for next_node in node["next"]:
            new_input = problem + f"前序输出: {response}\n"
            execute_node(next_node, new_input, problem)

    starts = [name for name in local_workflow if not any(name in v["next"] for v in local_workflow.values())]

    for start in starts:
        execute_node(start, problem, problem)

    app_type = detect_task_type(problem)

    final_outputs = ""
    for name, node in local_workflow.items():
        if "end" in node.get("next", []):
            outputs = node.get("outputs", [])
            if outputs:
                final_outputs = extract_module_code(outputs)[-1]
            else:
                final_outputs = f"Warning: Node '{name}' has no outputs."
            break

    code_quality = evaluate_code_quality(app_type, final_outputs)
    output_dir=f"./results/{code_name}/{method}/{iter_num}/output"
    output_file_path = os.path.join(output_dir, f"{i}_final_outputs.txt")
    with open(output_file_path, "w", encoding="utf-8") as f :
        f.write(final_outputs)
    print(f"✅ {i}: Final outputs saved to {output_file_path}")
    
    return {
        "iter_index": i,
        "final_outputs": final_outputs,
        "code_quality": code_quality,
        "total_cost_info": {
            "input_token": str(total_input_token),
            "output_token": str(total_output_token),
            "time": str(total_time_cost)
        },
        "node_cost_info": node_costs
    }
    
def run_workflow(workflow, code_name, problem, run_iters, iter_num=1, method="chatiot"):
    
    run_iters = 1
    results_dir = f"./results/{code_name}/{method}/{iter_num}"
    output_dir = f"./results/{code_name}/{method}/{iter_num}/output"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "workflow_results.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    print(f"\n--- Processing Task: {code_name} ---")

    # 获取 workflow_config（dict），而不是 workflow（含 LLM 实例）
    workflow_config = {
        name: {
            "type": node["type"],
            "params": node["params"],
            "next": node["next"]
        } for name, node in workflow.items()
    }

    # 并行评估代码成功率
    # args_list = [(process_id, workflow_config, problem, code_name, method, iter_num) for process_id in range(run_iters)]
    # with Pool(processes=run_iters) as pool:
    #     results_list = pool.map(execute_single_run, args_list)
    #     pool.close()
    #     pool.join()
    
    #禁用并行，防止和进化算个体并行评估的冲突
    results_list = []
    for process_id in range(run_iters):
        result = execute_single_run((process_id, workflow_config, problem, code_name, method, iter_num))
        results_list.append(result)

    app_type = detect_task_type(problem)

    code_quality_all = sum(r["code_quality"] for r in results_list)
    total_input_token_all = sum(float(r["total_cost_info"]["input_token"]) for r in results_list)
    total_output_token_all = sum(float(r["total_cost_info"]["output_token"]) for r in results_list)
    total_time_cost_all = sum(float(r["total_cost_info"]["time"]) for r in results_list)

    node_costs = results_list[-1]["node_cost_info"]

    task_result = {
        "task": code_name,
        "app_type": app_type,
        "workflow_topology": {
            name: {
                "type": node["type"],
                "params": node["params"],
                "outputs": node["outputs"],
                "next": node["next"]
            } for name, node in workflow.items()
        },
        "final_outputs": results_list[-1]["final_outputs"],
        "code_quality": code_quality_all / run_iters,
        "total_cost_info": {
            "input_token": str(total_input_token_all / run_iters),
            "output_token": str(total_output_token_all / run_iters),
            "time": str(total_time_cost_all / run_iters)
        },
        "node_cost_info": node_costs,
        "results_per_iter": results_list
    }

    results[code_name] = task_result

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Results for '{code_name}' saved to {json_path}")

    return results  
def run_predefined_workflow_config(filepath, benchmark_tasks, baseline="chatiot"):
    """Run a predefined workflow configuration from a JSON file."""
    
    workflow_config = load_workflow_config(filepath)
    workflow_config = workflow_config[baseline]
    workflow = build_workflow_with_config(workflow_config)
    for name, node in workflow.items():
        print(f"{name} params: {node['params']}, next: {node['next']}")
    iter_num = 1
    for code_name, problem in zip(data['code_name'], data["problem"]):
        if code_name not in benchmark_tasks:
            continue
        
        results =  run_workflow(workflow, code_name, problem, 5, iter_num, baseline)
        print(f"Results for {code_name}: {results}")
    
def LLM_meta_search_for_each(benchmark_tasks, max_iters=10):
    """
    Generates workflow configurations for each problem in the benchmark tasks using an LLM,
    selects the most frequently occurring configuration, and saves it to a local JSON file.
    
    Args:
        benchmark_tasks (list): List of task names to evaluate.
        output_path (str): Path to save the final selected workflow configuration.
        
    Returns:
        dict: The selected most frequent workflow configuration.
    """
    llm_workflow = get_llm(model="gpt-4o", temperature=0.5)
    search_space = create_operator_search_space()
    workflow_configs = []
    method = "meta_search"

    for type_name, problem in zip(data['code_name'], data["problem"]):
        
        if type_name not in benchmark_tasks:
            continue
        
        iter_num = 0
        all_attempts = []
        
        while iter_num <= max_iters:
            
            iter_num += 1   
            
            print(f"Iteration {iter_num} for {type_name}")
        
            system_prompt = f"""
                You are a programming assistant that solves problems step by step.
            """

            user_query = f"""
                Generate a workflow to solve the following problem: {problem}.
                Prioritize accuracy while minimizing token usage and inference time.
                Based on the complexity of the task '{type_name}', select specific operators from the search space :
                {search_space}, and select the certain params, [1,5] means that you can choose 1, 2, 3, 4 or 5. 
                Based on previous attempts:{all_attempts}, you should generate a new workflow that 
                can improve the reward.
                You can follow the steps below to improve a workflow:
                1. Check the quality of the previous workflow.
                2. If the quality > 0.9, you can reduce the number of operators in the workflow or decrease parameter values (for example, lower the num_agents in ensemble, reduce the max_iterations in reflexion, or choose a smaller model).
                3. If the quality < 0.9, you can add more operators to the workflow or increase parameter values (for example, increase the num_agents in ensemble, increase the max_iterations in reflexion, or choose a larger model).
                Then, you should compose them into an optimal workflow 
                to solve the given problem. You only need to output the workflow in json format.

                Your output must strictly follow the following format, you can not contain ```json  in your output:
                For example:
                {{
                    "cot": {{
                        "type": "cot",
                        "params": {{"model": "gpt-4o", "temperature": 0.5}},
                        "next": ["reflexion"]
                    }},
                    "reflexion": {{
                        "type": "reflexion",
                        "params": {{"model": "gpt-4o", "temperature": 0.5}},
                        "next": ["end"]
                    }}
                }}
                You do not need to include the above example exactly — feel free to generate a different structure based on the problem.
                """

            messages = [SystemMessage(system_prompt), HumanMessage(user_query)]
            print(f"Generating workflow for problem: {type_name}")
            response = llm_workflow.invoke(messages)
            print(f"Response from LLM for {type_name}: {response.content }")

            try:
                workflow_config = json.loads(response.content)
                workflow_configs.append(workflow_config)
                print(f"✅ Successfully generated workflow for {type_name}")
                
                with open(f"./results/{type_name}.json", "w") as f:
                    json.dump(workflow_config, f, indent=4)
                    
                # execute workflows
                workflow = workflow_configs[-1]
                for name, node in workflow.items():
                    print(f"{name} params: {node['params']}, next: {node['next']}")
                workflow = build_workflow_with_config(workflow)
                results = run_workflow(workflow, type_name, problem, 5, iter_num, method="meta_search")
                
                multi_rewards, token_cost, time_cost, code_quality = get_multi_objective_rewards(results, type_name)
                
                all_attempts.append({
                    "task": type_name,
                    "iteration": iter_num,
                    "config": workflow_config,
                    "rewards": multi_rewards,
                    "token_cost": token_cost,
                    "time_cost": time_cost, 
                    "code_quality": code_quality
                })

                # save history of all attempts
                history_attempts_dir =  f"./results/{type_name}/{method}"
                os.makedirs(history_attempts_dir, exist_ok=True)
                history_attempts_path = os.path.join(history_attempts_dir, "history_attempts.json")
                with open(history_attempts_path, "w") as f:
                    json.dump(all_attempts, f, indent=4)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse workflow for {type_name}: {e}")
                continue
        
    if not workflow_configs:
        raise ValueError("No valid workflow configurations were generated.")

def evaluate_individual(args):
    """
    单个个体的评估函数，用于并行执行。
    """
    individual, code_name, problem, method, generation_idx, individual_idx, population_size = args
    
    temp_config_dir = f"./results/{method}"
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_config_dir, f"temp_individual_{generation_idx}_{individual_idx}.json")
    
    results = []
    try:
        with open(temp_config_path, "w") as f:
            json.dump({"temp_config": individual}, f, indent=4)

        workflow_config = load_workflow_config(temp_config_path)["temp_config"]
        workflow = build_workflow_with_config(workflow_config)

        results = run_workflow(workflow, code_name, problem, 5, generation_idx * population_size + individual_idx + 1, method="genetic")
        
        return results 
    except Exception as e:
        print(f"Error evaluating individual {individual_idx}: {e}")
        return results  
def genetic_search_for_each(benchmark_tasks, max_iters=20):
    """
    Genetic algorithm to evolve workflow configurations with objectives:
    - Minimize token cost and time
    - Maximize code quality

    Args:
        benchmark_tasks (list): List of task names to evaluate.
        max_iters (int): Maximum number of generations for the genetic algorithm.

    Returns:
        dict: The best workflow configuration found.
    """
    population_size = 5
    mutation_rate = 0.2
    tournament_size = 2
    elite_size = 1
    max_operators = 3
    method = "genetic"

    search_space = create_operator_search_space()

    def random_operator():
        op_name = random.choice(list(search_space.keys()))
        
        params = {k: random.choice(v) if isinstance(v, list) else v for k, v in search_space[op_name].items()}
        return {
            "type": op_name,
            "params": params,
            "next": ["end"]
        }

    def generate_random_workflow():
        num_ops = random.randint(1, max_operators)
        ops = [random_operator() for _ in range(num_ops)]
        config = {}

        names = [f"op{i}" for i in range(num_ops)]
        for i, name in enumerate(names):
            config[name] = ops[i]
            config[name]["type"] = ops[i]["type"]

        G = nx.DiGraph()
        G.add_nodes_from(names + ["end"])
        for i in range(len(names)):
            if i < len(names) - 1:
                config[names[i]]["next"] = [names[i+1]]
        config[names[-1]]["next"] = ["end"]
        
        return config

    # Initialize population
    population = [generate_random_workflow() for _ in range(population_size)]

    best_config = None
    best_reward = float('-inf')
    
    for code_name, problem in zip(data['code_name'], data["problem"]):
        
        if code_name not in benchmark_tasks:
            continue
        
        generation_results = {} 
        
        for generation in range(0, max_iters):
            print(f"\n--- Generation {generation + 1} ---")

            # Evaluate fitness
            fitness_scores = []
            
            args_list = [(individual, code_name, problem, method, generation, individual_idx, population_size) for individual_idx, individual in enumerate(population)]
            with Pool(processes=population_size) as pool:
                results_list = pool.map(evaluate_individual, args_list)
                pool.close()
                pool.join()

            for idx, results in enumerate(results_list):
                if not results: 
                    print(f"⚠️ No results for individual {idx} in generation {generation + 1}")
                    continue
                task_results = {}
                total_reward = 0
                count = 0
                        
                reward, _, _, _ = get_multi_objective_rewards(results, code_name)
                
                task_results[code_name] = results[code_name]
                total_reward += reward
                count += 1

                avg_reward = total_reward / count if count > 0 else total_reward
                fitness_scores.append(avg_reward)

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_config = population[idx]
                        
                task_results[code_name]['reward'] = avg_reward
                generation_results[f"individual_{generation * population_size + idx + 1}"] = task_results
                 
            print("Fitness scores:", fitness_scores)
            # Save generation results
            evolution_dir = f"./results/{code_name}/{method}"
            draw_all_individuals_topology(generation_results, evolution_dir, 0)  
            os.makedirs(evolution_dir, exist_ok=True)
            gen_log_path = os.path.join(evolution_dir, f"generation_{generation + 1}.json")
            with open(gen_log_path, "w", encoding="utf-8") as f:
                json.dump(generation_results, f, indent=4, ensure_ascii=False)
            # Selection
            def tournament_selection():
                indices = random.sample(range(len(population)), tournament_size)
                best_idx = max(indices, key=lambda i: fitness_scores[i])
                return population[best_idx]

            new_population = []

            # Elitism
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
            elites = [population[i] for i in elite_indices]
            new_population.extend(elites)

            # Crossover and Mutation
            while len(new_population) < population_size:
                parent1 = tournament_selection()
                parent2 = tournament_selection()

                child = {}
                keys1, keys2 = list(parent1.keys()), list(parent2.keys())
                split_point = min(len(keys1), len(keys2)) // 2

                for i, key in enumerate(keys1):
                    if i < split_point:
                        child[key] = parent1[key]
                    else:
                        break

                for i, key in enumerate(keys2):
                    if i >= split_point:
                        child[key] = parent2[key]
                    else:
                        break
                
                if child:
                    keys = list(child.keys())
                    for i in range(len(keys) - 1):
                        child[keys[i]]["next"] = [keys[i+1]]
                    child[keys[-1]]["next"] = ["end"]

                if random.random() < mutation_rate:
                    mutation_type = random.choice(['param', 'add', 'remove'])

                    if mutation_type == 'param' and child:
                        key = random.choice(list(child.keys()))
                        op_type = child[key]["type"]
                        param_key = random.choice(list(search_space[op_type].keys()))
                        options = search_space[op_type][param_key]
                        value = random.choice(options) if isinstance(options, list) else options
                        child[key]["params"][param_key] = value

                    elif mutation_type == 'add' and len(child) < max_operators:
                        new_op = random_operator()
                        new_key = f"op{len(child)}"
                        child[new_key] = new_op
                        if len(child) > 1:
                            prev_key = random.choice(list(child.keys())[:-1])
                            child[prev_key]["next"] = [new_key]
                            child[new_key]["next"] = ["end"]

                    elif mutation_type == 'remove' and len(child) > 1:
                        key_to_remove = random.choice(list(child.keys())[:-1])
                        del child[key_to_remove]
                        keys = list(child.keys())
                        for i in range(len(keys) - 1):
                            child[keys[i]]["next"] = [keys[i+1]]
                        child[keys[-1]]["next"] = ["end"]

                new_population.append(child)

            population = new_population

    print("\n✅ Best workflow configuration found:")
    return best_config
    
def get_multi_objective_rewards(data, code_name):
    """
    Calculate rewards based on the results of the workflow execution.
    """
    
    app_type = data[code_name]["app_type"]
    cost_info = data[code_name]["total_cost_info"]
    final_outputs = data[code_name]["final_outputs"]
    input_tokens = float(cost_info["input_token"])
    output_tokens = float(cost_info["output_token"])
    time_cost = float(cost_info["time"])
    code_quality = data[code_name]["code_quality"]

    
    rewards = {
        "token_cost": input_tokens + output_tokens, 
        "time_cost":  time_cost,  
        "code_quality": code_quality 
    }
    
    print(f"Rewards: {rewards}")
    
    weights = {"token_cost": -0.4, "time_cost": -0.3, "code_quality": 0.3}
    
    reward = 0
    for key in rewards:
        reward += weights[key] * rewards[key]
        
    return reward, rewards["token_cost"], rewards["time_cost"], rewards["code_quality"]

def evaluate_code_quality(app_type, output_text):
    
    if app_type == "IFTTT":
        try:
            parsed = json.loads(output_text)
            
            required_keys = ["Thought", "Say_to_user", "Action_type"]
            for key in required_keys:
                if key not in parsed:
                    print(f"Missing required key: {key}")
                    return False, None
            return 1
    
        except json.JSONDecodeError as e:
            print(f"Invalid IFTTT format: {e}")
            return 0
    
    if app_type == "RIOT":
        
        from config import RIOT_ROOT
        workplace = os.path.join(RIOT_ROOT,'examples','LLM_Gen')
        filepath = os.path.join(RIOT_ROOT,'examples','LLM_Gen','main.c')
        # Write the code to the file
        try:
            with open(filepath,'w') as f:
                f.write(output_text)
        except Exception as e:
            return 0
        
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
                return 0
            else:
                return 1
        except subprocess.TimeoutExpired:
            return 0
    
    if app_type == "Python_executer":
        
        import tempfile
        import subprocess
        
        python_interpreter = "/home/hao/miniconda3/envs/python_executor/bin/python"
        if not os.path.exists(python_interpreter):
            print(f"Python interpreter not found at {python_interpreter}")
            return 0
        
        try:
            # Create a temporary file to store the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(output_text)
            
            # Execute the code with the specified Python interpreter and capture output
            # Using shell=True to simulate terminal execution and capture all output
            process = subprocess.Popen(
                f"{python_interpreter} {temp_filename}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
            )
            stdout, stderr = process.communicate(timeout=10)  
            os.unlink(temp_filename)
            
            if stderr:
                return 0
            else:
                return 1
                
        except subprocess.TimeoutExpired:
            # Handle timeout
            try:
                os.unlink(temp_filename)
            except:
                pass
            return 0
            
        except Exception as e:
            # Handle other exceptions
            try:
                os.unlink(temp_filename)
            except:
                pass
            return 0
    
def main():

    # Create search space
    
    search_space = create_operator_search_space()
    
    print("Search space defined:", search_space)
    
    # run_predefined_workflow_config("workflow_config.json", benchmark, "IoTPilot")
    
    # LLM_meta_search_for_each(benchmark, 10)
    
    genetic_search_for_each(benchmark, 10)
    

if __name__ == "__main__":
    main()