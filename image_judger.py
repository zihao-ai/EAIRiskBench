import argparse
import base64
import os

from openai import OpenAI
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def evaluate_image_with_text(img_path, scene, text_observation):

    img_file = img_path.split("/")[-1]

    img = Image.open(img_path)

    width, height = img.size
    img1 = img.crop((0, 0, width // 2, height // 2))
    img2 = img.crop((width // 2, 0, width, height // 2))
    img3 = img.crop((0, height // 2, width // 2, height))
    img4 = img.crop((width // 2, height // 2, width, height))

    img_dir = f"results/scene_img/{scene}/segmented_images"
    os.makedirs(img_dir, exist_ok=True)
    img1_path = os.path.join(img_dir, f"{img_file}_1.png")
    img2_path = os.path.join(img_dir, f"{img_file}_2.png")
    img3_path = os.path.join(img_dir, f"{img_file}_3.png")
    img4_path = os.path.join(img_dir, f"{img_file}_4.png")

    img_path_list = [img1_path, img2_path, img3_path, img4_path]

    img1.save(img1_path)
    img2.save(img2_path)
    img3.save(img3_path)
    img4.save(img4_path)

    prompt = f"""Your task is to evaluate the compatibility between the image and the caption.  You will be given four images about the scene with index 1-4,  and a natural language caption about the scene. 
    You need to carefully analyze each image and determine which one best matches the caption provided. Consider aspects such as objects, object relationships described in the caption when making your judgment. Just output the index of the best matching image. Do not include other information in the output.
    [Caption:] {text_observation}"""

    content = [{"type": "text", "text": prompt}]

    for img_path_i in img_path_list:
        base64img = encode_image(img_path_i)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64img}"}})

    client = OpenAI(api_key="")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=1000,
        temperature=0,
    )

    match_idx = int(response.choices[0].message.content) - 1
    matched_img_path = img_path_list[match_idx]

    return matched_img_path


def main():
    # add argparse process_num
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--text_observation", type=str)
    args = parser.parse_args()

    matched_img_path = evaluate_image_with_text(
        img_path=args.img_path, scene=args.scene, text_observation=args.text_observation
    )
    print(matched_img_path)


if __name__ == "__main__":
    main()
