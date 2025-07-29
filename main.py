import os
import base64
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
      
endpoint = os.getenv("ENDPOINT_URL", "https://<REPLACE_ENDPOINT_NAME>.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")
      
# Initialize Azure OpenAI client with Entra ID authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
    default_headers={"x-ms-oai-ev3-predictor_search_length": "4"}
)

completion = client.chat.completions.with_raw_response.create(
    model=deployment,
    messages=[
        {"role": "user", "content": "1 + 1"},
    ],
    prediction={"type": "content", "content": "1 + 1 = 2"},
    max_tokens=800,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False
)
print(completion)
print(completion.text)
print(completion.headers)
print(completion.headers['x-request-id'])
print(completion.headers['apim-request-id'])