import argparse
import json
import os

import pandas as pd

from plan_evaluation import plan_execution_evaluation, plan_safety_evaluation
from plan_generation_with_image import plan_generation_with_image
from plan_generation_with_text import plan_generation_with_text
from utils.llm_util import GPT, llm_response_to_json
from utils.skill_set import skill_set, skill_set_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="EAIRiskDataset/dataset.csv")
    parser.add_argument("--img_folder", type=str, default="EAIRiskDataset/images")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    args = parser.parse_args()

    result_path = f"results/{args.model}.json"

    results = {}

    df = pd.read_csv(args.dataset_path, index_col=False, skipinitialspace=True, escapechar="\\", quotechar='"')

    for index, row in df.iterrows():
        id = row["ID"]
        scene = row["Scene"]
        safety_tip = row["Safety Tip"]
        explanation = row["Tip Explanation"]
        instruction = row["Instruction"]
        objects = row["Objects"]
        objects_positions = row["Objects Positions"]
        objects_attributes = row["Objects Attributes"]
        text_observation = row["Text Observation"]
        image_observation = row["Matched Image Path"]
        image_path = f"{args.img_folder}/{image_observation}"

        llm_model = GPT(api_key=args.api_key, api_url=args.api_url, model=args.model)

        plans_with_text = plan_generation_with_text(
            scene=scene, textual_observation=text_observation, instruction=instruction, skill_set=skill_set_string, model=llm_model
        )

        plans_with_text_execution_evaluation = plan_execution_evaluation(
            skill_set=skill_set,
            plans=plans_with_text,
        )

        plans_with_text_safety_evaluation = plan_safety_evaluation(
            safety_tip=safety_tip,
            explanation=explanation,
            plans=plans_with_text,
            observation=text_observation,
            model=llm_model,
        )

        plans_with_image = plan_generation_with_image(
            scene=scene, image_path=image_path, instruction=instruction, objects=objects, skill_set=skill_set_string, model=llm_model
        )

        plans_with_image_execution_evaluation = plan_execution_evaluation(
            skill_set=skill_set,
            plans=plans_with_image,
        )

        plans_with_image_safety_evaluation = plan_safety_evaluation(
            safety_tip=safety_tip,
            explanation=explanation,
            plans=plans_with_image,
            observation=text_observation,
            model=llm_model,
        )

        if scene not in results:
            results[scene] = {}
        results[scene][id] = {}
        results[scene][id]["safety tip"] = safety_tip
        results[scene][id]["tip explanation"] = explanation
        results[scene][id]["instruction"] = instruction
        results[scene][id]["text observation"] = text_observation
        results[scene][id]["image observation"] = image_path
        results[scene][id]["plans_with_text"] = {
            "plans": plans_with_text,
            "safety evaluation": plans_with_text_safety_evaluation,
            "execution evaluation": plans_with_text_execution_evaluation,
        }
        results[scene][id]["plans_with_image"] = {
            "plans": plans_with_image,
            "safety evaluation": plans_with_image_safety_evaluation,
            "execution evaluation": plans_with_image_execution_evaluation,
        }
        print(f"Success: {index} {scene} {id}")

    with open(result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
