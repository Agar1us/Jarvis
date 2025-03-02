import yaml
from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel
from tools import *
import importlib

def test_screenshot():
    custom_tools = ScreenShotTool()
    custom_tools.forward('Is the Telegram application open on the computer?')


def test_agents():
    model = OpenAIServerModel(
        model_id='deepseek-chat',
        api_base='https://api.deepseek.com',
        api_key='',
        temperature=0.0,
    )

    prompt = yaml.safe_load(
            importlib.resources.files("prompts").joinpath("computer_usage_prompt.yaml").read_text()
        )
    
    computer_agent = CodeAgent(
        tools=[UserInputTool(), ComputerInputTool(), ScreenShotTool()],
        model=model,
        name="computer_agent",
        prompt_templates=prompt,
    )
    computer_agent.run(listen_and_transcribe())


if __name__ == "__main__":
    test_agents()
