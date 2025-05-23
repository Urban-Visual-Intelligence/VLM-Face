"""An example showing how to use vLLM to serve multimodal models 
and run online inference with OpenAI client.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve llava-hf/llava-1.5-7b-hf --chat-template template_llava.jinja

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
    --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2

(audio inference with Ultravox)
vllm serve fixie-ai/ultravox-v0_3 --max-model-len 4096
"""
import base64
import time
import os
import json
import requests
import argparse

from openai import OpenAI

# from vllm.assets.audio import AudioAsset
# from vllm.utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Multi-image input inference
def run_multi_image(model, prompt, image1, image2, temperature=0.1):

    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": "You are a helpful assistant."
            },
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                'type': 'image_url',
                'image_url': {
                    'url':
                    f"data:image/jpeg;base64,{image1}",
                    },
                },
                {
                'type': 'image_url',
                'image_url': {
                    'url':
                    f"data:image/jpeg;base64,{image2}",
                    },
                },

            ],
        }],
        model=model,
        max_tokens=4096,
        temperature=temperature
    )

    result = chat_completion_from_url.choices[0].message.content
    return result


def parse_pairs_txt(file_path):
    pairs = []

    # Read the file
    with open(file_path, 'r') as f:
        for line in f:
            # Split line into components
            parts = line.strip().split()
            if len(parts) == 3:
                image1, image2, label = parts
                pairs.append({
                    'image1': image1,
                    'image2': image2,
                    'label': int(label)  # Convert label to integer
                })
            elif len(parts) == 5:
                image1, image2, label, age1, age2 = parts
                pairs.append({
                    'image1': image1,
                    'image2': image2,
                    'label': int(label),  # Convert label to integer
                    'age1': int(float(age1)),
                    'age2': int(float(age2))
                })
    
    return pairs


question = """
Task: Compare two face images and determine whether they belong to the same person or different people.
Instruction: 
1. Do not give the conclusion in the beginning.
2. When comparing the two images, prioritize facial features that are non-variant or less affected by changes such as lighting, facial expressions, camera angle, or age.
3. Based on the information above, you need to generate a comparison of facial features between the two images. During the comparison, you must specify the locations of the compared features in both images.
4. Based on the comparsion, provide the answer with Yes or No.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple Python script with arguments.")

    # Add arguments
    parser.add_argument("-r", "--result", type=str, help="result path", required=True)
    parser.add_argument("-a", "--api", type=str, help="api address", required=True)
    parser.add_argument("-c", "--cot", type=int, default=0, help="cot or direct", required=True)
    parser.add_argument("-t", "--temp", type=float, default=0, help="temperature", required=False)

    # Parse arguments
    args = parser.parse_args()
    file_path = "/mnt/afs/machengqian/data/face/cp_lfw/pairs_CPLFW.txt"
    image_folder_path = "/mnt/afs/machengqian/data/face/cp_lfw/images"
    result_path = args.result
    # result_path = "/mnt/afs/machengqian/mllm/qwen2vl/results/cplfw_qwen2_5vl_7b_grpo_0313_0_kl_cot_v1_t_0.json"
    # model_path = "/home/mnt/share_team2/share_model/Qwen/Qwen2-VL-72B-Instruct"
    # model = "/home/mnt/share_team2/share_model/Qwen/Qwen2-VL-7B-Instruct"
    
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    openai_api_key = "EMPTY"
    openai_api_base = args.api
    # openai_api_base = "http://localhost:8000/v1"
    # openai_api_base = "http://localhost:6060/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id
    print(model)
    # QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer with only Yes or No in <answer> </answer> tags."
    QUESTION_TEMPLATE = "{Question} Output Format: Based on the Instruction above, output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

    if args.cot:
        question = QUESTION_TEMPLATE.format(Question=question)    
    result = dict()
    pairs = parse_pairs_txt(file_path)

    for pair in pairs:
        start = time.time()
        image1 = encode_image(os.path.join(image_folder_path, pair["image1"]))
        image2 = encode_image(os.path.join(image_folder_path, pair["image2"]))

        response = run_multi_image(model, question, image1, image2, args.temp)

        print(response)
        if pair["label"] == 1:
            result[f"{pair['image1']}_{pair['image2']}"] = [response, "Yes"]
        else:
            result[f"{pair['image1']}_{pair['image2']}"] = [response, "No"]
        end = time.time()
        print("inference speed: ", end - start)
    
        with open(result_path, 'w') as file:
        # Write text to the file
            json.dump(result, file, indent=4, ensure_ascii=False)
