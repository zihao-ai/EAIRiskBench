import argparse
import time

import requests



def submit_midjounery_task(prompt, api_key):
    url = "https://api.turboai.one/mj-relax/mj/submit/imagine"

    headers = {
        "content-type": "application/json",
        "mj-api-secret": api_key,
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    }

    data = {
        "base64Array": [],
        "notifyHook": "",
        "prompt": f"{prompt} --no lens blur --version 6  --quality 1  --chaos 0  --stylize 0 --aspect 4:3",
        "state": "",
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def request_task_status(task_id, api_key):
    url = f"https://api.turboai.one/mj/task/{task_id}/fetch"
    headers = {
        "content-type": "application/json",
        "mj-api-secret": api_key,
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    }
    response = requests.get(url, headers=headers)
    return response.json()


def save_img_from_url(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as image_file:
            image_file.write(response.content)
        return {"success": True}
    else:
        return {"error": response.text}


def midjourney_img_generation(prompt, file_path, api_key):
    timeout = 300
    start_time = time.time()

    response = submit_midjounery_task(prompt, api_key)

    if response["code"] == 24:
        bannded_word = response["properties"]["bannedWord"]
        prompt = prompt.replace(bannded_word, "")
        response = submit_midjounery_task(prompt, api_key)
    elif response["code"] != 1:
        print(f"Failed to submit task: {response['description']}")
        return {"error": "Failed to submit task"}

    if response["code"] == 1:
        task_id = response["result"]
        print(f"Task {task_id} submitted.")

        time.sleep(30)

        while True:
            if time.time() - start_time > timeout:
                return {"error": "Task timed out"}
            status_response = request_task_status(task_id, api_key)
            status = status_response["status"]
            progress = status_response["progress"]

            if status == "SUCCESS" and progress == "100%":
                image_url = status_response["imageUrl"]
                try:
                    save_img_from_url(image_url, file_path)
                except Exception as e:
                    return {"error": str(e)}
                else:
                    return {"success": True, "file_path": file_path}
            elif status == "FAILURE":
                return {"error": status_response["failReason"]}
            else:
                progress = status_response["progress"]
                print(f"Task {task_id}: {progress}")
                time.sleep(10)
    else:
        return {"error": "Failed to submit task"}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()
    midjourney_img_generation(prompt=args.prompt, file_path=args.output, api_key=args.api_key)


if __name__ == "__main__":
    main()
