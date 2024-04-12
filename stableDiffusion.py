import boto3
import json
import base64
import os


Prompt = '''
   generate image of a chinese castle with cherry trees.
'''

bedrock = boto3.client(service_name = "bedrock-runtime")

payload = {
    "text_prompts" : [{"text" : Prompt , "weight": 1}],
    "cfg_scale" : 10,
    "seed" : 0,
    "steps" : 50,
    "width" : 1024,
    "height" : 1024
}

body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-v1"

response = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response["body"].read())

print(response_body)

artifact = response_body["artifacts"][0]
image_encoded = artifact["base64"].encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)