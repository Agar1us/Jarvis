from typing import Optional, List

from smolagents.tools import Tool
from smolagents.models import Model
from omniparser.model import make_screenshot


DESCRIPTION = """
...
Use this tool when you need to analyze a screenshot or get a interactive element.
It takes user queries as input and generates detailed answers about the screenshot, including the identification and location of interactive elements (e.g., buttons, links, input fields). 

Examples:

Example 1:
    >>> data = "You need to make sure that the Telegram application exists on your computer."
    >>> answer = screenshot_extraction(data=data)
    >>> print(answer)
    >>> Yes, the Telegram application is on the computer and you can interact with it. Coordinates of the interactive element: 1400, 740

Example 2:
    >>> data = "Is there a verification window?"
    >>> answer = screenshot_extraction(data=data)
    >>> print(answer)
    >>> Yes, this is the verification window for the Yandex Music app.
"""

MODEL_PROMPT = """
You are a computer assistant. Your task is to answer questions related to screenshots that the user asks.
He may ask if there is an object in the image, and if so, it is necessary to specify its coordinates, he may ask what is in the image itself, for example, whether we have entered an application or a web page, etc.
You will be given a screenshot, as well as all its interactive elements (the text of the elements and their coordinates), so that you can correctly answer the question.
The screenshot will show the boxes and their numbers, and each box will have a description in the form of text.

Example 1:
User question: Is there a search bar in the picture?
Interactive elements:  [{'content': 'Show all tabs', 'coordinates': [1748, 0]}, {'content': 'Heart', 'coordinates': [846, 974]}, {'content': 'Close', 'coordinates': [1877, 0]}, {'content': 'a forward or next action.', 'coordinates': [1228, 461]}, {'content': 'Time', 'coordinates': [1664, 453]}, {'content': 'A video streaming application.', 'coordinates': [819, 31]}, {'content': 'a back or move action.', 'coordinates': [337, 461]}, {'content': 'Shuffle', 'coordinates': [1624, 974]}, {'content': 'Navigation Bar', 'coordinates': [1045, 4]}, {'content': 'Highlight', 'coordinates': [1575, 976]}, {'content': 'New Project', 'coordinates': [150, 996]}, {'content': 'Time Field', 'coordinates': [1665, 371]}, {'content': '2, September, 2024', 'coordinates': [1585, 815]}, {'content': 'Delete', 'coordinates': [2, 985]}, {'content': 'Lemonade', 'coordinates': [1667, 853]}, {'content': 'Time Field', 'coordinates': [1181, 591]}, {'content': 'Office...', 'coordinates': [800, 971]}, {'content': 'Cloud', 'coordinates': [5, 1033]}]
Output: Yes, there is a search bar in the image, coordinates: 1045, 4

Example 2
User question: Have you logged into the Yandex Music app?
Interactive elements: [{'content': 'Hysteria', 'coordinates': [1329, 939]}, {'content': 'Kroww', 'coordinates': [1399, 939]}, {'content': '5*14', 'coordinates': [1723, 939]}, {'content': 'Ripples in the Sand from Dune   Offic', 'coordinates': [610, 974]}, {'content': 'на > ~', 'coordinates': [1576, 982]}, {'content': 'White Stork Hans Zimmer', 'coordinates': [611, 997]}, {'content': '8 C', 'coordinates': [53, 1043]}, {'content': '23.12', 'coordinates': [1867, 1041]}, {'content': 'Облачно', 'coordinates': [49, 1057]}, {'content': '20,02.2025', 'coordinates': [1839, 1057]}, {'content': 'Dirty', 'coordinates': [1329, 855]}, {'content': 'Microsoft Word', 'coordinates': [0, 293]}, {'content': 'My', 'coordinates': [7, 33]}, {'content': 'Microsoft 365', 'coordinates': [98, 32]}, {'content': 'A purple and black image viewer.', 'coordinates': [358, 30]}, {'content': 'Ticket', 'coordinates': [45, 0]}, {'content': 'Navigator', 'coordinates': [1048, 1037]}, {'content': 'Draw Functions', 'coordinates': [9, 0]}]
Output: Based on the image, we went to the Yandex music application page.

--Real data--\n
"""


class ScreenShotTool(Tool):
    name = 'screenshot_extraction'
    description = DESCRIPTION
    inputs = {
        "data": {
            "type": "string",
            "description": "Full text from the user to analyze the information on the screen",
            "nullable": True,
        }
    }
    output_type = "string"

    def forward(
        self,
        data: Optional[str] = None
    ) -> str:
        image, descriptions = make_screenshot()
        question = f"User question: {data}\nInteractive elements: {descriptions}\nOutput:"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MODEL_PROMPT + question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ]
        from openai import OpenAI
        client = OpenAI(api_key='key')
        completion = client.chat.completions.create(model='gpt-4o-mini', messages=messages, temperature=0.0)
        answer = completion.choices[0].message.content
        return answer
