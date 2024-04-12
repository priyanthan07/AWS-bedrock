import boto3
import json

Prompt = '''
    write a summary regarding the future of the Gen AI
'''

bedrock = boto3.client(service_name = "bedrock-runtime")

payload = {
    "prompt" : "[INST]" + Prompt + "[/INST]",
    "max_gen_len" : 512,
    "temperature" : 0.5,
    "top_p" : 0.9
}

body = json.dumps(payload)
model_id = "meta.llama2-70b-chat-v1"

response = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response["body"].read())
response_text = response_body["generation"]

print(response_text)

#  The generated text is
output = '''
        1. Increased Adoption: Gen AI is likely to become more ubiquitous in various industries such as healthcare, finance, education, and transportation, among others. As the technology improves, more businesses and organizations are likely to adopt Gen AI solutions to streamline processes, improve efficiency, and reduce costs.
        2. Expansion of Application Areas: Gen AI will increasingly be applied to new areas such as climate change, agriculture, and urban planning. For instance, AI-powered systems can be used to monitor and manage natural resources, predict and prevent natural disasters, and optimize urban planning and infrastructure development.
        3. Continued Advances in Deep Learning: Deep learning, a subset of machine learning, is expected to continue to drive advancements in Gen AI. Deep learning algorithms will become more sophisticated and capable of solving complex problems, leading to breakthroughs in areas such as natural language processing, computer vision, and robotics.
        4. Increased Focus on Ethics and Transparency: As Gen AI becomes more pervasive, there will be a growing need for ethical considerations around its development and deployment. Developers and users of Gen AI will need to ensure that the technology is transparent, fair, and unbiased, and that it does not perpetuate existing social inequalities or create new ones.
        5. Human-AI Collaboration: Gen AI will increasingly be used to augment human capabilities, rather than replace them. Humans and AI systems will work together to solve complex problems and create new products and services. This collaboration will require developers to design AI systems that can effectively communicate and collaborate with humans.
        6. Concerns around Job Displacement: There is a concern that Gen AI could displace human jobs, particularly in industries where repetitive tasks are common. This could lead to significant social and economic impacts, and will require proactive measures to mitigate the effects of job displacement.
        7. Ensuring Safety and Security: As Gen AI becomes more powerful, there is

'''