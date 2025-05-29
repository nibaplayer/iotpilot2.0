import json
import matplotlib.pyplot as plt
import itertools

# 支持的颜色和标记
colors_genetic = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
markers_genetic = ['o', 's', '^', 'P', '*', 'X', 'D', 'v']

# 读取遗传算法数据
file_path_genetic = "/home/hao/code/iotpilot2.0/src/results/ControlRoom/genetic/generation_1.json"
with open(file_path_genetic, "r") as file:
    data_genetic = json.load(file)

# 初始化存储列表：遗传算法数据
code_qualities_genetic = []
input_tokens_genetic = []
output_tokens_genetic = []
times_genetic = []
individual_labels_genetic = []

for individual_key in data_genetic.keys():
    print(individual_key)
    individual = data_genetic[individual_key]["ControlRoom"]
    code_quality = individual["code_quality"]
    total_cost_info = individual["total_cost_info"]

    input_token = float(total_cost_info["input_token"])
    output_token = float(total_cost_info["output_token"])
    time = float(total_cost_info["time"])

    # 添加到列表
    code_qualities_genetic.append(code_quality)
    input_tokens_genetic.append(input_token)
    output_tokens_genetic.append(output_token)
    times_genetic.append(time)
    individual_labels_genetic.append(individual_key)

# 读取chatiot数据
file_path_chatiot = "/home/hao/code/iotpilot2.0/src/results/ControlRoom/chatiot/1/workflow_results.json"
with open(file_path_chatiot, "r") as file:
    data_chatiot = json.load(file)

# 读取iotpilot数据
file_path_chatiot = "/home/hao/code/iotpilot2.0/src/results/ControlRoom/IoTPilot/1/workflow_results.json"
with open(file_path_chatiot, "r") as file:
    IoTPilot_IoTPilot = json.load(file)

# 提取 chatiot 数据
motion_data = data_chatiot["ControlRoom"]
code_quality_chatiot = motion_data["code_quality"]
total_cost_info = motion_data["total_cost_info"]

input_token_chatiot = float(total_cost_info["input_token"])
output_token_chatiot = float(total_cost_info["output_token"])
time_chatiot = float(total_cost_info["time"])

# 提取 IoTPilot 数据
motion_data_iotpilot = IoTPilot_IoTPilot["ControlRoom"]
code_quality_iotpilot = motion_data_iotpilot["code_quality"]
total_cost_info_iotpilot = motion_data_iotpilot["total_cost_info"]

input_token_iotpilot = float(total_cost_info_iotpilot["input_token"])
output_token_iotpilot = float(total_cost_info_iotpilot["output_token"])
time_iotpilot = float(total_cost_info_iotpilot["time"])

plt.rcParams.update({
    'font.size': 18,               # 基础字体大小
    'axes.titlesize': 22,          # 子图标题
    'axes.labelsize': 16,          # 横轴/纵轴标签
    'xtick.labelsize': 16,         # x轴刻度标签
    'ytick.labelsize': 16,         # y轴刻度标签
    'legend.fontsize': 18,         # 图例字体
    'legend.title_fontsize': 16   # 图例标题字体
})

# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
point_size = 200

# 为遗传算法分配颜色和形状
color_cycle_gen = itertools.cycle(colors_genetic)
marker_cycle_gen = itertools.cycle(markers_genetic)

individual_color_map = {}
individual_marker_map = {}

for label in individual_labels_genetic:
    if label not in individual_color_map:
        individual_color_map[label] = next(color_cycle_gen)
    if label not in individual_marker_map:
        individual_marker_map[label] = next(marker_cycle_gen)

# 绘制遗传算法数据
for i, label in enumerate(individual_labels_genetic):
    color = individual_color_map[label]
    marker = individual_marker_map[label]

    axes[0].scatter(code_qualities_genetic[i], input_tokens_genetic[i],
                    color=color, marker=marker, s=point_size,
                    edgecolor='black', label=label)

    axes[1].scatter(code_qualities_genetic[i], output_tokens_genetic[i],
                    color=color, marker=marker, s=point_size,
                    edgecolor='black', label=label)

    axes[2].scatter(code_qualities_genetic[i], times_genetic[i],
                    color=color, marker=marker, s=point_size,
                    edgecolor='black', label=label)

# 绘制chatiot数据（单独一个点）
chatiot_label = "chatiot"
chatiot_color = 'black'
chatiot_marker = 'o'

axes[0].scatter(code_quality_chatiot, input_token_chatiot,
                color=chatiot_color, marker=chatiot_marker, s=point_size * 1.5,
                edgecolor='black', label=chatiot_label, zorder=10)

axes[1].scatter(code_quality_chatiot, output_token_chatiot,
                color=chatiot_color, marker=chatiot_marker, s=point_size * 1.5,
                edgecolor='black', label=chatiot_label, zorder=10)

axes[2].scatter(code_quality_chatiot, time_chatiot,
                color=chatiot_color, marker=chatiot_marker, s=point_size * 1.5,
                edgecolor='black', label=chatiot_label, zorder=10)

# 绘制 IoTPilot 数据（单独一个点）
iotpilot_label = "IoTPilot"
iotpilot_color = 'darkorange'
iotpilot_marker = 'D'

axes[0].scatter(code_quality_iotpilot, input_token_iotpilot,
                color=iotpilot_color, marker=iotpilot_marker, s=point_size * 1.5,
                edgecolor='black', label=iotpilot_label, zorder=10)

axes[1].scatter(code_quality_iotpilot, output_token_iotpilot,
                color=iotpilot_color, marker=iotpilot_marker, s=point_size * 1.5,
                edgecolor='black', label=iotpilot_label, zorder=10)

axes[2].scatter(code_quality_iotpilot, time_iotpilot,
                color=iotpilot_color, marker=iotpilot_marker, s=point_size * 1.5,
                edgecolor='black', label=iotpilot_label, zorder=10)


# 设置标题和标签
axes[0].set_xlabel('Code Quality')
axes[0].set_ylabel('Input Token')
axes[0].set_title('Input Token vs Code Quality')

axes[1].set_xlabel('Code Quality')
axes[1].set_ylabel('Output Token')
axes[1].set_title('Output Token vs Code Quality')

axes[2].set_xlabel('Code Quality')
axes[2].set_ylabel('Time (seconds)')
axes[2].set_title('Time vs Code Quality')

# 去重图例
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# 在绘制完所有点后，更新标签
new_labels = []
for label in labels:
    if label.startswith("individual_"):
        new_labels.append(f"operator #{label.split('_')[1]}")
    else:
        new_labels.append(label)

# 更新图例
fig.legend(by_label.values(), new_labels,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.15),
           ncol=5)

plt.tight_layout()
plt.savefig('motivation_python.pdf', dpi=1000, bbox_inches='tight')