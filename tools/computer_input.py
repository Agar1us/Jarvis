from smolagents.tools import Tool  # type: ignore
from smolagents.models import Model  # type: ignore
import pyautogui

# Set pause between actions for stability
pyautogui.PAUSE = 0.1

def execute_operation(input_string):
    """
    Executes a mouse or keyboard operation based on the input string.
    
    Input string format:
    - For mouse: "mouse operation data" (e.g., "mouse moveTo 100 200" or "mouse click left")
    - For keyboard: "keyboard operation data" (e.g., "keyboard type Hello World" or "keyboard hotkey ctrl c")
    
    Args:
        input_string (str): Input string with operation type, operation, and data.
    
    Returns:
        str: Result of the operation execution or error message.
    """
    try:
        # Split the input string into parts
        parts = input_string.strip().split(maxsplit=2)
        if len(parts) < 2:
            return "Error: insufficient arguments. Format: 'type operation data'"

        operation_type = parts[0].lower()  # mouse or keyboard
        operation = parts[1].lower()       # operation (click, type, moveTo, etc.)
        data = parts[2] if len(parts) > 2 else ""  # data (coordinates or text/commands)

        # Check operation type
        if operation_type == "mouse":
            return handle_mouse_operation(operation, data)
        elif operation_type == "keyboard":
            return handle_keyboard_operation(operation, data)
        else:
            return f"Error: unknown operation type '{operation_type}'. Valid types: mouse, keyboard"

    except Exception as e:
        return f"Execution error: {str(e)}"

def handle_mouse_operation(operation, data):
    """
    Handles mouse operations.
    
    Args:
        operation (str): Operation (click, moveTo, etc.)
        data (str): Data (coordinates or click type)
    
    Returns:
        str: Result of the operation execution.
    """
    if operation == "moveto":
        try:
            x, y = map(int, data.split())
            pyautogui.moveTo(x, y, duration=0.5)
            return f"Mouse moved to coordinates ({x}, {y})"
        except ValueError:
            return "Error: moveTo requires two numeric coordinates (e.g., '100 200')"

    elif operation == "click":
        button = data.lower() if data else "left"  # Default is left click
        if button not in ["left", "right", "middle"]:
            return "Error: unknown mouse button. Valid options: left, right, middle"
        pyautogui.click(button=button)
        return f"Mouse click performed (button: {button})"

    else:
        return f"Error: unknown mouse operation '{operation}'. Valid operations: moveTo, click"

def handle_keyboard_operation(operation, data):
    """
    Handles keyboard operations.
    
    Args:
        operation (str): Operation (type, press, hotkey)
        data (str): Data (text or commands)
    
    Returns:
        str: Result of the operation execution.
    """
    if operation == "type":
        if not data:
            return "Error: type requires text to input"
        pyautogui.write(data)
        return f"Text entered: '{data}'"

    elif operation == "press":
        if not data:
            return "Error: press requires a key"
        pyautogui.press(data)
        return f"Key pressed: '{data}'"

    elif operation == "hotkey":
        if not data:
            return "Error: hotkey requires a key combination"
        keys = data.split(' + ')
        pyautogui.hotkey(*keys)
        return f"Key combination performed: {keys}"

    else:
        return f"Error: unknown keyboard operation '{operation}'. Valid operations: type, press, hotkey"
    

DESCRIPTION="""
Use this tool when you need to perform mouse or keyboard actions to interact with the user interface. It allows you to automate mouse movements, clicks, typing text, and keyboard shortcuts.
The function takes a string command that specifies the operation type (mouse or keyboard), the specific operation, and any required data.
Command format: "{{operation_type}} {{operation}} {{data}}"

Mouse operations:

moveTo: Moves cursor to specified coordinates (requires x y coordinates)
click: Performs mouse click (optional button: left, right, middle)

Keyboard operations:

type: Types the specified text
press: Presses a single key
hotkey: Performs key combinations (space-separated keys)

Examples:

Example 1:
result = computer_input(data="mouse moveTo 100 200")
print(result)
'Mouse moved to coordinates (100, 200)'

Example 2:
result = computer_input(data="mouse click left")
print(result)
'Mouse click performed (button: left)'

Example 3:
result = computer_input(data="keyboard type Hello World")
print(result)
'Text entered: 'Hello World''

Example 4:
result = computer_input(data="keyboard hotkey ctrl + c")
print(result)
'Key combination performed: ['ctrl', 'c']'
"""


class ComputerInputTool(Tool):
    name = 'computer_input'
    description = DESCRIPTION
    inputs = {
    "data": {
        "type": "string",
        "description": "Command string specifying operation type, operation, and data. Format: '{{operation_type}} {{operation}} {{data}}'",
        }
    }
    output_type = "string"

    def forward(self, data: str) -> str:
        return execute_operation(data)
