from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import json
import os
from enum import Enum
from loguru import logger
from types import SimpleNamespace

load_dotenv()
# Define the OpenAI API key and model
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = 'gpt-4o'

# Define the function schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        }
    }
]

# Simulate a user prompt
user_prompt = "What is the weather like in San Francisco, CA?"
#user_prompt = "Hello, good morning"

# Generate the model's response
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_prompt}],
    tools=tools,
    tool_choice="auto"

)

message = response.choices[0].message
finish_reason = response.choices[0].finish_reason
print(finish_reason)
print(message)

if "tool_calls" in finish_reason:
    tool_call = message.tool_calls[0]
    function=tool_call.function
    function_name=function.name
    function_arguments = json.loads(function.arguments)

    print(f"tool call: {tool_call}")
    print(f"function name: {function_name}")
    print(f"function arguments:{function_arguments}")


# # Check if the model wants to call a function
# if "tool_calls" in response.choices[0].finish_reason:
#     function_call = response.choices[0].message["function_call"]
#     function_name = function_call["name"]
#     arguments = function_call["arguments"]

#     # Simulate calling the function (replace with actual API call)
#     if function_name == "get_weather":
#         location = arguments.get("location")
#         # Example: get weather data using some weather API
#         weather_data = {
#             "temperature": "68°F",
#             "condition": "Sunny"
#         }

#         # Pass the function result back to the model for further conversation
#         function_response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": user_prompt},
#                 {"role": "assistant", "function_call": function_call},
#                 {"role": "function", "name": function_name, "content": str(weather_data)}
#             ]
#         )

#         # Print the final response
#         print(function_response.choices[0].message["content"])
