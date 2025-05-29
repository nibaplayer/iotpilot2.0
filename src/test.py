import os
import json
from utils import draw_all_individuals_topology  # Import the function to be tested

def create_test_data():
    """Create sample generation results data."""
    return {
        "individual_0": {
        "motion_sensor_80": {
            "task": "motion_sensor_80",
            "app_type": "IFTTT",
            "workflow_topology": {
                "op0": {
                    "type": "cot",
                    "params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0
                    },
                    "outputs": "null",
                    "next": [
                        "op1"
                    ]
                },
                "op1": {
                    "type": "cotsc",
                    "params": {
                        "model": "gpt-4o",
                        "temperature": 1,
                        "N": 5
                    },
                    "outputs": "null",
                    "next": [
                        "end"
                    ]
                }
            },
            "final_outputs": "{\n    \"Thought\": \"The TAP has been generated using the bathroom's motion sensor to trigger the action of turning on the bedroom light and setting the brightness to 80%. The motion sensor's state is checked for 'true' and the actions are to turn on the light and set the brightness accordingly.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true, 3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
            "code_quality": 1.0,
            "total_cost_info": {
                "input_token": "42145.0",
                "output_token": "5045.0",
                "time": "56.05570578575134"
            },
            "node_cost_info": {
                "op0": {
                    "input_token": "4786.0",
                    "output_token": "693.0",
                    "time": "7.240491151809692"
                },
                "op1": {
                    "input_token": "37359.0",
                    "output_token": "4352.0",
                    "time": "48.81521463394165"
                }
            },
            "results_per_iter": [
                {
                    "iter_index": 0,
                    "final_outputs": "{\n    \"Thought\": \"The TAP has been generated using the bathroom's motion sensor to trigger the action of turning on the bedroom light and setting the brightness to 80%. The motion sensor's state is checked for 'true' and the actions are to turn on the light and set the brightness accordingly.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true, 3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
                    "code_quality": 1,
                    "total_cost_info": {
                        "input_token": "42145.0",
                        "output_token": "5045.0",
                        "time": "56.05570578575134"
                    },
                    "node_cost_info": {
                        "op0": {
                            "input_token": "4786.0",
                            "output_token": "693.0",
                            "time": "7.240491151809692"
                        },
                        "op1": {
                            "input_token": "37359.0",
                            "output_token": "4352.0",
                            "time": "48.81521463394165"
                        }
                    }
                }
            ],
            "reward": -18892.516711735727
        }
    },
    "individual_1": {
        "motion_sensor_80": {
            "task": "motion_sensor_80",
            "app_type": "IFTTT",
            "workflow_topology": {
                "op0": {
                    "type": "cotsc",
                    "params": {
                        "model": "gpt-4o",
                        "temperature": 1,
                        "N": 5
                    },
                    "outputs": "null",
                    "next": [
                        "op1"
                    ]
                },
                "op1": {
                    "type": "reflexion",
                    "params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "max_iterations": 1
                    },
                    "outputs": "null",
                    "next": [
                        "end"
                    ]
                }
            },
            "final_outputs": "{\n    \"Thought\": \"Based on the user request, the trigger is 'motion state of the bathroom's motion sensor is true', and the action is 'turn on the bedroom light and set the brightness to 80%'. For the trigger, the device is the motion sensor in the bathroom with id 2. The service is motion-sensor. The property is motion-state and the state is true. For the action, the device is the light in the bedroom with id 3. The service is light. The properties are on with state true and brightness with state 80%.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true,3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
            "code_quality": 1.0,
            "total_cost_info": {
                "input_token": "50368.0",
                "output_token": "5743.0",
                "time": "79.95130324363708"
            },
            "node_cost_info": {
                "op0": {
                    "input_token": "32576.0",
                    "output_token": "4196.0",
                    "time": "49.43402981758118"
                },
                "op1": {
                    "input_token": "17792.0",
                    "output_token": "1547.0",
                    "time": "30.517273426055908"
                }
            },
            "results_per_iter": [
                {
                    "iter_index": 0,
                    "final_outputs": "{\n    \"Thought\": \"Based on the user request, the trigger is 'motion state of the bathroom's motion sensor is true', and the action is 'turn on the bedroom light and set the brightness to 80%'. For the trigger, the device is the motion sensor in the bathroom with id 2. The service is motion-sensor. The property is motion-state and the state is true. For the action, the device is the light in the bedroom with id 3. The service is light. The properties are on with state true and brightness with state 80%.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true,3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
                    "code_quality": 1,
                    "total_cost_info": {
                        "input_token": "50368.0",
                        "output_token": "5743.0",
                        "time": "79.95130324363708"
                    },
                    "node_cost_info": {
                        "op0": {
                            "input_token": "32576.0",
                            "output_token": "4196.0",
                            "time": "49.43402981758118"
                        },
                        "op1": {
                            "input_token": "17792.0",
                            "output_token": "1547.0",
                            "time": "30.517273426055908"
                        }
                    }
                }
            ],
            "reward": -22468.085390973094
        }
    },
    "individual_2": {
        "motion_sensor_80": {
            "task": "motion_sensor_80",
            "app_type": "IFTTT",
            "workflow_topology": {
                "op0": {
                    "type": "reflexion",
                    "params": {
                        "model": "gpt-4o",
                        "temperature": 0,
                        "max_iterations": 1
                    },
                    "outputs": "null",
                    "next": [
                        "end"
                    ]
                }
            },
            "final_outputs": "{\n    \"Thought\": \"Based on the user request, the trigger is the motion state of the bathroom's motion sensor being true. The corresponding device is motion-sensor in the bathroom with ID 2. The service is 'motion-sensor' and the property is 'motion-state'. The action consists of turning on the bedroom light and setting its brightness to 80%. The device is the light in the bedroom with ID 3, using service 'light', properties 'on' and 'brightness'.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true,3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
            "code_quality": 1.0,
            "total_cost_info": {
                "input_token": "15362.0",
                "output_token": "1354.0",
                "time": "30.599732875823975"
            },
            "node_cost_info": {
                "op0": {
                    "input_token": "15362.0",
                    "output_token": "1354.0",
                    "time": "30.599732875823975"
                }
            },
            "results_per_iter": [
                {
                    "iter_index": 0,
                    "final_outputs": "{\n    \"Thought\": \"Based on the user request, the trigger is the motion state of the bathroom's motion sensor being true. The corresponding device is motion-sensor in the bathroom with ID 2. The service is 'motion-sensor' and the property is 'motion-state'. The action consists of turning on the bedroom light and setting its brightness to 80%. The device is the light in the bedroom with ID 3, using service 'light', properties 'on' and 'brightness'.\",\n    \"TAP\": {\n        \"trigger\": \"2.motion-sensor.motion-state==true\",\n        \"condition\": \"\",\n        \"action\": \"3.light.on=true,3.light.brightness=80\"\n    },\n    \"Say_to_user\": \"Ok, I have generated the TAP for you.\",\n    \"Action_type\": \"Finish\"\n}",
                    "code_quality": 1,
                    "total_cost_info": {
                        "input_token": "15362.0",
                        "output_token": "1354.0",
                        "time": "30.599732875823975"
                    },
                    "node_cost_info": {
                        "op0": {
                            "input_token": "15362.0",
                            "output_token": "1354.0",
                            "time": "30.599732875823975"
                        }
                    }
                }
            ],
            "reward": -6695.279919862747
        }
    }
    }

def run_test():
    """Run the test for draw_all_individuals_topology function."""
    # Create test data
    generation_results = create_test_data()
    
    # Define the evolution directory where the topology will be saved
    evolution_dir = "./test_evolution"
    
    # Define the generation number (for testing purposes)
    generation = 0
    
    # Call the function to be tested
    draw_all_individuals_topology(generation_results, evolution_dir, generation)
    
    # Check if the file was created successfully
    expected_file_path = os.path.join(evolution_dir, f"topology_{generation + 1}.pdf")
    if os.path.exists(expected_file_path):
        print(f"Test passed! Topology saved to {expected_file_path}")
    else:
        print("Test failed! Topology file not found.")

if __name__ == "__main__":
    # Ensure the test_evolution directory exists
    os.makedirs("./test_evolution", exist_ok=True)
    
    # Run the test
    run_test()