import base64
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_resize(image_path, target_width=1024):
    with Image.open(image_path) as img:
        original_width, original_height = img.size

        new_width = 1024
        new_height = int((new_width / original_width) * original_height)

        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        buffered = BytesIO()
        resized_img.save(buffered, format="png")

        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64


class GPT:
    def __init__(self, api_key, api_url=None, model="gpt-4o-2024-08-06"):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    def generate(
        self,
        user_prompt,
        response_format=None,
        system_prompt="You are a helpful assistent.",
        temperature=0.7,
        max_tokens=4096,
    ):
        if self.api_url is None:
            client = OpenAI(api_key=self.api_key)
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate_with_image(
        self, image_path, user_prompt, response_format=None, system_prompt="You are a helpful assistent.", temperature=0.7, max_tokens=4096
    ):
        base64_image = encode_image(image_path)
        client = OpenAI(api_key=self.api_key, base_url=f"{self.api_url}")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

