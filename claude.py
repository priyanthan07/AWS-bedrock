import boto3
import json

Prompt = '''
    write a summary regarding the future of the Gen AI
'''

bedrock = boto3.client(service_name = "bedrock-runtime")

payload = {
    "prompt" :  Prompt,
    "max_tokens" : 512,
    "temperature" : 0.9,
    "top_p" : 0.9
}

body = json.dumps(payload)
model_id = "ai21.j2-mid-v1"

response = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response["body"].read())
response_text = response_body["completion"][0]["data"]["text"]

print(response_text)