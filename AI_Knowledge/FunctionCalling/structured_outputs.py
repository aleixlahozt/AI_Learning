from openai import OpenAI, AzureOpenAI, pydantic_function_tool
from dotenv import load_dotenv
import json
import os
from enum import Enum
from loguru import logger
from types import SimpleNamespace
from pydantic import BaseModel, Field
import time
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = 'gpt-4o'

from pydantic import BaseModel, Field
from typing import Optional

class TransferToAgentWithDataGathering(BaseModel):
    license_plate: str = Field(...,description="Vehicle registration plate. Default is 'ABCD123'")
    contract_number: str = Field(...,description="The client contract number. Could contain letters and digits, but convert all numbers into digits. Default is '1234567'")
    address: str = Field(...,description="The caller address. Default is 'Carrer Diputació 23'")
    name: str = Field(...,description="The full name of the caller. Set as 'Aleix Lahoz' if not provided") #TRICK TO SET DEFAULT
    age: Optional[int] = Field(description="The age of the caller")
    gender: str = Field(None, description="The gender of the caller")
    zodiac_sign: Optional[str] = Field(None, description="The zodiac sign of the caller")

def test_structured_outputs():

    system_prompt = """
    You are a helpful and professional call center agent specializing in assisting clients who are experiencing issues on the road. 
    Your main goal is to collect the necessary information from the client and then transfer them to the appropriate agent for further assistance.

    Follow these guidelines:
    1. Always gather all the required data before proceeding with the transfer.
    2. Be polite, clear, and concise in your responses, guiding the user step by step to provide the necessary details.
    3. Confirm the correctness of any important information (like license plate, contract number, address, and name) by repeating it back to the user.
    4. If a user asks for help, ensure that you collect the following information:
    - License plate number.
    - Contract number (which could contain letters and digits).
    - The caller’s address.
    - The full name of the caller.
    - Age.
    - Gender.
    - Zodiac Sign
    5. Only proceed to transfer the client once all the required information has been confirmed and collected.
    6. If at any point you are unable to assist, or if the client requests to speak directly to an agent, gather the necessary data and transfer them to the agent.
    7. Remain calm, empathetic, and professional throughout the conversation.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello, I'm having trouble with my vehicle, and I think I need to talk to an agent. Can you help?"},
        {"role": "assistant", "content": "Of course! Before I transfer you to an agent, I'll need to gather a few details to ensure we can assist you properly. Let’s start with your vehicle's registration plate. Could you please provide me with your license plate number?"},
        {"role": "user", "content": "Sure, it’s ABC12345."},
        {"role": "assistant", "content": "Thank you for that! Next, I’ll need your contract number, which might contain letters and digits. Could you please share that with me?"},
        {"role": "user", "content": "Yeah, it's CON4567."},
        {"role": "assistant", "content": "Great, thank you! Now, I’ll need your address to ensure we have your correct location on file. Could you provide that for me, please?"},
        {"role": "user", "content": "Sure, I live at 123 Main Street, Springfield."},
        {"role": "assistant", "content": "Thank you! Finally, may I have your full name? I’ll spell it back to you for confirmation once I have it."},
        #{"role": "user", "content": "My name is John Doe."},
        #{"role": "assistant", "content": "Thank you, John. Your name is spelled as J-O-H-N D-O-E. Is that correct?"},
        #{"role": "user", "content": "Yes, that’s correct."}
    ]

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=TransferToAgentWithDataGathering
    )

    print("\n###### S.O. Response: #######\n")
    print(completion.choices[0].message.parsed)

    print("\n###### How Tool JSON Schema would look if used for Function Calling: #######")
    tool_definition = pydantic_function_tool(model=TransferToAgentWithDataGathering, 
                                             name="transfer_to_agent_with_data_gathering", 
                                             description="Trigger when the user asks to be transferred to the agent")
    print(json.dumps(tool_definition, indent=4))

if __name__ == "__main__":
    test_structured_outputs()