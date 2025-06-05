from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing import Union
from config import *
import re
import json
import os

import paramiko
import json
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

class myllm:
    def __init__(self, model: str, temperature: float = 0.5, **kwargs):
        self.model = model
        self.temperature = temperature
        self.ssh_host = "10.214.149.209"
        self.ssh_port = 14000
        self.ssh_username = "root"
        self.ssh_password = "wop"
        self.api_endpoint = "http://localhost:11434/api/generate"

        

    def invoke(self, messages):
        import json
        import shlex

        # Step 1: Convert messages to string and safely escape for JSON
        combined_prompt = str(messages)
        safe_prompt = json.dumps(combined_prompt, ensure_ascii=False)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(
                hostname=self.ssh_host,
                port=self.ssh_port,
                username=self.ssh_username,
                password=self.ssh_password
            )

            # 构造原始命令
            raw_data = f'{{"model": "{self.model}", "prompt": {safe_prompt}, "stream": false}}'

            # 使用 shlex.quote 确保整个字符串在 shell 中安全
            safe_data = shlex.quote(raw_data)

            # 构造最终 curl 命令
            curl_command = f"curl --location --request POST '{self.api_endpoint}' " \
                            f"--header 'Content-Type: application/json' " \
                            f"--data-raw {safe_data}"

            stdin, stdout, stderr = ssh.exec_command(curl_command)

            output = stdout.read().decode()
            error = stderr.read().decode()
            try:
                response_json = json.loads(output)
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                return ""

            response = response_json.get("response", "")
            return response

        finally:
            ssh.close()


def get_llm(model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
    if model not in CANDIATE_MODEL:
        raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
    elif model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))
    elif model in ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:13b"]:
        llm = myllm(model=model, temperature=temperature)
    return llm

def extract_module_code(text: str) -> list:
    """
    Extract all code blocks wrapped in triple backticks (any language).
    E.g., ```c, ```python, etc.

    Args:
        text (str): The input text containing potential code blocks.

    Returns:
        List[str]: A list of strings representing the extracted code blocks.
    """
    pattern = r"```[a-zA-Z+]*\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def load_workflow_config(workflow_config_file_path: str) -> dict:
    """Load workflow config from json file and return a list of WorkflowConfig namedtuples."""
    if os.path.exists(workflow_config_file_path):
        with open(workflow_config_file_path, "r", encoding="utf-8") as f:
            workflow_config = json.load(f)
        print("Workflow config loaded from file:", workflow_config_file_path)
    else:
        raise FileNotFoundError(f"配置文件 '{workflow_config_file_path}' 未找到！")
    return workflow_config


import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

def draw_all_individuals_topology(code_name, generation_results, evolution_dir, generation):
    
    """
    绘制每个 individual 的 workflow_topology, 每个 individual 单独作为一个 subfigure。
    
    参数:
        generation_results (dict): 包含个体结果的数据，格式为 {"individual_0": {...}, ...}
        evolution_dir (str): 保存图像的目录路径
        generation (int): 当前代数，用于命名输出文件
    """
    individuals = {k: v for k, v in generation_results.items() if k.startswith("individual_")}
    num_individuals = len(individuals)

    # 创建画布和子图
    max_cols = 5
    num_individuals = len(individuals)

    # 自动计算行数和列数
    nrows = (num_individuals + max_cols - 1) // max_cols
    ncols = min(max_cols, num_individuals)

    figsize_width = ncols * 4
    figsize_height = nrows * 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height))
    if num_individuals == 1:
        axes = [axes]  # 确保 axes 是列表以便后续迭代
    else:
        axes = axes.flatten().tolist()  # 将多维数组展平为一维列表

    # 定义不同 operator 类型的颜色映射
    color_map = {
        "cot": "lightblue",
        "cotsc": "lightgreen",
        "reflexion": "lightcoral",
        "debate": "lightskyblue",
        "ensemble": "lightyellow"
    }

    for idx, (individual_name, individual_data) in enumerate(individuals.items()):
        ax = axes[idx]
        topo = individual_data.get(code_name, {}).get('workflow_topology', {})
        cost_info = individual_data.get(code_name, {}).get('total_cost_info', {})
        code_quality = individual_data.get(code_name, {}).get('code_quality', 0)


        G = nx.DiGraph()
        edge_list = []

        for node, info in topo.items():
            if node == "end":
                continue

            op_type = info["type"]
            params = info["params"]

            filtered_params = {k: v for k, v in params.items() if k != "temperature"}
            node_color = color_map.get(op_type, "lightgray")

            clean_node_name = node.replace("op", "")
            label = f"{op_type}\n{json.dumps(filtered_params, indent=1)}"
            G.add_node(clean_node_name, label=label, color=node_color)

            for next_node in info.get("next", []):
                if next_node != "end":
                    clean_next_node_name = next_node.replace("op", "")
                    edge_list.append((clean_node_name, clean_next_node_name))

        for src, dst in edge_list:
            G.add_edge(src, dst)

        from networkx.drawing.nx_agraph import to_agraph

        A = to_agraph(G)
        A.layout(prog='dot')
        pos = {}
        for node in A.nodes():
            if hasattr(node, 'attr') and 'pos' in node.attr:
                x, y = map(float, node.attr['pos'].split(','))
                pos[node] = (x, y)
        labels = nx.get_node_attributes(G, 'label')
        for node in G.nodes:
            if 'color' not in G.nodes[node]:
                G.nodes[node]['color'] = "lightgray"
        colors = [G.nodes[node]['color'] for node in G.nodes]

        # 绘图
        if pos:
            nx.draw(G, pos, with_labels=True, labels=labels,
                    node_size=1500, node_color=colors, font_size=5,
                    font_weight="bold", arrows=True, alpha=0.9, ax=ax)

        input_tk = cost_info.get("input_token", "N/A")
        output_tk = cost_info.get("output_token", "N/A")
        time_cost = cost_info.get("time", "N/A")

        ax.set_title(
            f"{individual_name}\n"
            f"Input: {input_tk}, Output: {output_tk}\n"
            f"Time: {time_cost}\n"
            f"code_quality: {code_quality}",
            fontsize=10, pad=20
        )

        ax.axis('on')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.)  

    os.makedirs(evolution_dir, exist_ok=True)
    save_topology = os.path.join(evolution_dir, f"topology_{generation + 1}.pdf")
    plt.savefig(save_topology, format='pdf', bbox_inches='tight')
    plt.close(fig)