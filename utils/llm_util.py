import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from llm_models.gpt import GPT



import json
import re
import ast

def llm_response_to_json(response):
    response = response.replace("\n", "")
    
    # Attempt to parse directly as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting content wrapped with ```json
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try extracting content wrapped with any ``` block
    code_block_pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        potential_json = match.group(1)
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

    # Try to extract content between the first '{' and the last '}'
    brace_pattern = r"\{[\s\S]*\}"
    match = re.search(brace_pattern, response)
    if match:
        json_str = match.group(0)
        try:
            # Attempt parsing with ast.literal_eval for JSON-like structures
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

    # Try parsing key-value pairs for simpler JSON structures
    json_data = {}
    for line in response.split(","):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            json_data[key] = value
    if json_data:
        return json_data
    
    # If all attempts fail, return None or raise an error
    raise ValueError(f"Could not parse response as JSON: {response}")




def llm_generate(model, user_prompt, system_prompt="You are a helpful assitent.", response_format=None):

    if response_format is None:
        response = model.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    else:
        response = model.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=response_format,
        )

    return response


def llm_generate_with_image(model, image_path, user_prompt, system_prompt=None, response_format=None):

    if response_format is None:
        response = model.generate_with_image(
            image_path=image_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    else:
        response = model.generate_with_image(
            image_path=image_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=response_format,
        )
    return response

