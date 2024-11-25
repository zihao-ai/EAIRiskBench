import argparse

from utils.llm_util import GPT


def text2img_prompt_generator(scene, objects, object_positions, object_attributes, model):
    prompt = f"""Your task is to generate a prompt for a text-to-image diffusion model, such as Midjourney. You are given some scene information, including a scene name, the objects in the scene, their positions and relationships, and their attributes.
You need to generate a prompt for the text-to-image diffusion model to generate scene images that is a high-quality, realistic simulation of the scene.
The prompt should make the image visually represent the scene, including all objects, their positions and relationships, and their attributes.
The prompt should be precise. Do not add added effects. The prompt should conform to Midjourney's prompt specification.

The input scene information is as follows:
[Scene]: {scene}
[Objects]: {objects}
[Object positions]: {object_positions}
[Object attributes]: {object_attributes}

The output should be a string that is a valid prompt for the text-to-image diffusion model. Just output the prompt, do not include other information.
"""
    response = model.generate(
        user_prompt=prompt,
        system_prompt="You are a helpful assistant to generate a prompt for a text-to-image diffusion model.",
        max_tokens=4096,
        response_format=None,
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--objects", type=str)
    parser.add_argument("--object_positions", type=str)
    parser.add_argument("--object_attributes", type=str)

    args = parser.parse_args()

    llm_model = GPT(api_key=args.api_key, api_url=args.api_url, model=args.model)
    prompt = text2img_prompt_generator(
        args.scene, args.objects, args.object_positions, args.object_attributes, llm_model
    )
    print(prompt)
    return prompt


if __name__ == "__main__":
    main()
