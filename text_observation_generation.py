import argparse

from utils.llm_util import GPT


def scene2observation(scene, objects, object_positions, object_attributes, model):
    prompt = f"""Your task is to generate text-based observation of the scene from the robot's egoview based on the given scene information. 
The observation should describe the scene using natural language in detail, including all objects, their positions and relationships, and their attributes. 
The observation is used for the robot to understand the environment with natural language to make high-level plans.

Here is an example:
You are in the center of a kitchen. There is a microwave_1 on the marble kitchen counter_1. The door of the microwave_1 is closed. Beside the kitchen counter_1, there is a wooden dining table_1. On the wooden dining table_1, there is a metal bowl_1 made of stainless steel and a sandwich_1.

Your Task:
The input scene information is as follows, where the numbers in the object names serve as unique identifiers to differentiate between multiple instances of similar objects:
[Scene]: {scene}
[Objects]: {objects}
[Object positions]: {object_positions}
[Object attributes]: {object_attributes}

The output should be a string that is a valid observation for the robot. Just output the observation, do not include other information.
"""
    response = model.generate(
        user_prompt=prompt,
        system_prompt="You are a helpful assistant to generate a text-based observation of the scene from the robot's egoview.",
        max_tokens=4096,
        response_format=None,
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--objects", type=str)
    parser.add_argument("--object_positions", type=str)
    parser.add_argument("--object_attributes", type=str)
    args = parser.parse_args()

    llm_model = GPT(api_key=args.api_key, api_url=args.api_url, model=args.model)
    observation = scene2observation(
        scene=args.scene,
        objects=args.objects,
        object_positions=args.object_positions,
        object_attributes=args.object_attributes,
        model=llm_model,
    )
    print(observation)
    return observation


if __name__ == "__main__":
    main()
