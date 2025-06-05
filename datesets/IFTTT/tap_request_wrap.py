import json
import csv

file_path_tap = "tap.csv"
device_list_path = "device.json"

def generate_tap_prompt(user_request, device_list):
    prompt = f"""
You are a useful assistant named TapGenerator in the field of smart home. Your task is to parse user input into a trigger-action program (TAP). A TAP consists of three parts: trigger, condition, and action. The trigger is the event that starts the automation, the condition is optional, and the action is what the system performs.

# Input
1. User request: {user_request}
2. Device list: {device_list}

# Workflow
1. Understand the user's request and extract the trigger, condition, and action.
2. Match devices and services from the list.
3. Generate the TAP in the required format.
4. Handle ambiguous or missing information by asking the user.

# Output Format
Return a JSON object with:
- Thought: Explanation of how the TAP was generated.
- Action_type: Either "AskUser" or "Finish".
- Say_to_user: Natural language response to the user.
- TAP: JSON structure {{ "trigger": "...", "condition": "...", "action": "..." }}

Examples:
{{
    "Thought": "Based on the user request...",
    "TAP": {{
        "trigger": "2.motion-sensor.motion-state==true",
        "condition": "",
        "action": "1.light.on=true, 1.light.brightness=80"
    }},
    "Say_to_user": "Ok, I have generated the TAP for you.",
    "Action_type": "Finish"
}}
"""
    return prompt 
def read_device_list(device_list_path):
    """Reads the device list from a JSON file."""
    try:
        with open(device_list_path, 'r') as file:
            device_list = json.load(file)
        return device_list
    except Exception as e:
        print(f"Error reading device list: {e}")
        return []


def read_user_request(file_path):
    """Reads user requests from a CSV file."""
    user_requests = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_requests.append(row)
    except Exception as e:
        print(f"Error reading user requests: {e}")
    return user_requests

def process_user_request():
    """Processes each user request and matches with the corresponding devices."""
    user_requests = read_user_request(file_path_tap)
    device_list = read_device_list(device_list_path)

    id = 0
    for request in user_requests:
        room_name = request.get('room_name', '').lower()
        request_value = request.get('request', '')

        matched_devices = [device for device in device_list if device['area'].lower() == room_name]
        
        if matched_devices:
            wrapped_request = generate_tap_prompt(request_value, matched_devices)
            with open('wrapped_tap.csv', mode='a', encoding='utf-8', newline='') as csvfile:
                codename = f"tap_{id}"
                writer = csv.writer(csvfile)
                writer.writerow([codename, wrapped_request])
                id += 1

process_user_request()