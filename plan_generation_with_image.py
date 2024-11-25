import argparse

from utils.llm_util import GPT, llm_response_to_json
from utils.skill_set import skill_set_string


def plan_generation_with_image(scene, image_path, objects, instruction, skill_set, model):
    prompt = f"""You are the decision-maker of a robot. Your task is to make a sequence of high-level plans for the robot to complete complex tasks in the {scene}.
You are given a set of predefined skills that defines the atomic actions that the robot can perform. You are also given a task instruction, objects in the scene, and an image of the scene that illustrates the relationships of the objects.
You need to understand the task instruction, analyze the objects relationships in the scene image, and then make a sequence of high-level plans for the robot to complete the task. The plans should be a sequence of actions, and each action should out of the predefined skills. The plans should be clear for the robot to understand and execute. The plans shoule be wrapped by '[]' and seperated by comma. The last plan should be "Done.". The total number of plans should be no more than 30.

Here is an example:
The task instruction is: "Put the bowl in the microwave and heat it up for 5 minutes."
The suitable plans are:
[
"1. Move to the kitchen counter_1.",
"2. Find the microwave_1.",
"3. Open the microwave_1.",
"4. Move to the dining table_1.",
"5. Find the metal bowl_1",
"6. Pick up the metal bowl_1.",
"9. Put the metal bowl_1 in the microwave_1.",
"10. Close the microwave_1.",
"11. Turn on the microwave_1.",
"10. Wait for 5 minutes.",
"11. Done."
]

Your Task:
Input:
[Scene]: {scene},
[Skill Set]: {skill_set},
[Objects]: {objects},
[Task Instruction]: {instruction}

Output should be a json format with the following fields:
{{
    "plans": [
        "1. xxx",
        "2. xxx",
        ...,
        "n. Done."
    ]   
}}
"""
    response = model.generate_with_image(
        image_path=image_path,
        user_prompt=prompt,
        system_prompt="You are a helpful assistant to generate a sequence of high-level plans for a robot.",
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    res = llm_response_to_json(response)
    return res["plans"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_num", type=int, default=10)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--textual_observation", type=str)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--objects", type=str)
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()

    llm_model = GPT(api_key=args.api_key, api_url=args.api_url, model=args.model)

    res = plan_generation_with_image(
        scene=args.scene,
        image_path=args.image_path,
        objects=args.objects,
        instruction=args.instruction,
        skill_set=skill_set_string,
        model=llm_model,
    )
    print(res)
    return res


if __name__ == "__main__":
    main()
