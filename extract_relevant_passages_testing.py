import time
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from prompt.extract_relevant_passages_prompt import system_prompt, user_prompt, prediction_content

endpoint = os.getenv("ENDPOINT_URL")
      
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


deployments = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
results_with_prediction = {}
results_without_prediction = {}

print("Testing latency for different GPT-4.1 models (WITH and WITHOUT prediction)...\n")

# Test with prediction
print("=" * 50)
print("TESTING WITH PREDICTION")
print("=" * 50)

for deployment in deployments:
    print(f"Testing {deployment} with prediction...")

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            prediction={"type": "content", "content": prediction_content},
        )
        end_time = time.time()
        
        latency = end_time - start_time
        results_with_prediction[deployment] = latency
        print(f"  ✓ {deployment}: {latency:.3f} seconds")

        # Print prediction token details
        if hasattr(completion, 'usage') and completion.usage and hasattr(completion.usage, 'completion_tokens_details'):
            details = completion.usage.completion_tokens_details
            print(f"    Prediction tokens - Accepted: {details.accepted_prediction_tokens}, Rejected: {details.rejected_prediction_tokens}")
        
    except Exception as e:
        print(f"  ✗ {deployment}: Error - {str(e)}")
        results_with_prediction[deployment] = None

    print()

# Test without prediction
print("=" * 50)
print("TESTING WITHOUT PREDICTION")
print("=" * 50)

for deployment in deployments:
    print(f"Testing {deployment} without prediction...")

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        end_time = time.time()
        
        latency = end_time - start_time
        results_without_prediction[deployment] = latency
        print(f"  ✓ {deployment}: {latency:.3f} seconds")
        
    except Exception as e:
        print(f"  ✗ {deployment}: Error - {str(e)}")
        results_without_prediction[deployment] = None

    print()

# Display results
print("=" * 60)
print("LATENCY COMPARISON RESULTS")
print("=" * 60)

print("\nWITH PREDICTION:")
print("-" * 30)
valid_with = {k: v for k, v in results_with_prediction.items() if v is not None}
if valid_with:
    for deployment in deployments:
        if deployment in valid_with:
            print(f"{deployment:<15}: {valid_with[deployment]:.3f} seconds")

print("\nWITHOUT PREDICTION:")
print("-" * 30)
valid_without = {k: v for k, v in results_without_prediction.items() if v is not None}
if valid_without:
    for deployment in deployments:
        if deployment in valid_without:
            print(f"{deployment:<15}: {valid_without[deployment]:.3f} seconds")

print("\nDIFFERENCE (Without - With):")
print("-" * 30)
for deployment in deployments:
    if deployment in valid_with and deployment in valid_without:
        diff = valid_without[deployment] - valid_with[deployment]
        print(f"{deployment:<15}: {diff:+.3f} seconds")

if valid_with and valid_without:
    print("\nSUMMARY:")
    print("-" * 30)
    
    fastest_with = min(valid_with, key=valid_with.get)
    fastest_without = min(valid_without, key=valid_without.get)
    
    print(f"Fastest with prediction:    {fastest_with} ({valid_with[fastest_with]:.3f}s)")
    print(f"Fastest without prediction: {fastest_without} ({valid_without[fastest_without]:.3f}s)")