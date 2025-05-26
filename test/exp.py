import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import json

from src.Operator import CoT, CoTSC, Reflexion, DebateOperator, EnsembleOperator

benchmark =[
    # "RIOT_MQTT",
    # "RIOT_COAP",
    # "RIOT_MQTT_SN",
    # "RIOT_IRQ",
    # "RIOT_RTC",
    # "RIOT_UDP",
    # "RIOT_Thread",
    # "RIOT_Timer",
    # "RIOT_Flash",
    # "RIOT_MMA",
    # "RIOT_Event",
    # "RIOT_DHT11",
    # "RIOT_Warn",
    # "RIOT_Sched",
    # "RIOT_MBOX",
    # "FreeRTOS_MQTT",
    # "FreeRTOS_COAP",
    # "FreeRTOS_UDP",
    # "FreeRTOS_FLASH",
    # "FreeRTOS_MMA",
    # "Zephyr_MQTT",
    # "Zephyr_COAP",
    # "Zephyr_IRQ",
    # "Zephyr_UDP",
    # "Zephyr_MMA",
    # "Contiki_DHT11",
    # "Contiki_led",
    # "Contiki_UDP",
    # "Contiki_COAP",
    # "Contiki_MQTT",
    
    "motion_sensor_80",
    "environment_air",
    "motion_sensor_20",
    "illumination_sensor",
    "motion_sensor_color_white",
    "motion_sensor_warm_white",
    "environment_heat_fan",
    
    # "IMU_HAR",
    # "Heartbeat_detection",
    # "ControlLED",
    # "ControlRoom",
    # "RecognizeFace",
    # "SensorFusion",
    # "DingTalkNotification",
    
]

if __name__ == '__main__':
    cot = CoT(model="gpt-4o", temperature=0.5)
    cotsc = CoTSC(model="gpt-4o", temperature=0.5)
    reflexion = Reflexion(model="gpt-4o", temperature=0.5)
    ensemble = EnsembleOperator(model="gpt-4o", temperature=0.5, num_agents=3)
    debate = DebateOperator(model="gpt-4o", temperature=0.5)
    operators = {"cot": cot, "cotsc": cotsc, "reflexion": reflexion, "ensemble": ensemble, "debate": debate}
    data = pd.read_csv("/home/hao/code/iotpilot2.0/datesets/app.csv")
    iterators = 1
    results_dir = "results"
    res = {}

    # Create results directory if not exists
    os.makedirs(results_dir, exist_ok=True)

    for op_name, op in operators.items():
        # Initialize result structure
        res[op_name] = {}

        # Create operator-specific directory
        op_dir = os.path.join(results_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)

        print(f"Processing data with {op_name} operator...")

        for type_name, problem in zip(data['code_name'], data["problem"]):
            
            if type_name not in benchmark:
                continue
            
            res[op_name][type_name] = []
            problem_dir = os.path.join(op_dir, type_name)
            print(f"Processing '{problem_dir}'...")
            os.makedirs(problem_dir, exist_ok=True)

            for i in range(iterators):
                print(f"Running {op_name} on '{type_name}', iteration {i+1}...")

                result = op.run(problem)
                print(f"Result for {problem}: {result}")

                # Save raw result to file
                result_file = os.path.join(problem_dir, f"result{i+1}.txt")
                with open(result_file, "w") as f:
                    f.write(result)
                    print(f"Saved result to {result_file}")

                # Record cost and reset
                res[op_name][type_name].append(op.get_cost())
                op.reset_cost()

                # Persist current state to JSON after each iteration
                json_path = os.path.join(results_dir, "results.json")
                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2)