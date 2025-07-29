import time
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://testaistudio0307049060.cognitiveservices.azure.com/",
    api_key="<API_KEY>",
    api_version="2025-01-01-preview",
)

system_prompt = """
You will be given a topic or question of interest, along with a markdown-formatted document.\n
Your task is to extract all passages from the markdown that are directly relevant to the given topic or question.\n\n
<instructions>\n
- Each passage must be copied exactly as it appears in the markdown, preserving all original content and formatting\n
- If multiple passages are relevant, separate them with newlines and an ellipsis (`\\\n...\\\n`)\n
- DO NOT modify or summarize the content in any way\n
- Output ONLY the extracted passages; DO NOT include any explanations or additional text\n
- If there are no relevant passages, return an empty result\n</instructions>
"""

user_prompt = """
<topic_or_question>\n
Summarize the latest AI news and developments.\n
</topic_or_question>\n\n
<markdown_document>\n
# Update from Amazon CEO Andy Jassy on Generative AI\n\n
# Message from CEO Andy Jassy: Some thoughts on Generative AI\n\n
Written by Andy Jassy, CEO of Amazon\n\nJune 17, 2025\n\n5 min read\n\n
The message below was shared with Amazon employees earlier today.\n\n
Today, in virtually every corner of the company, we're using Generative AI to make customers lives better and easier. 
What started as deep conviction that every customer experience would be reinvented using AI, and that altogether new experiences we've only dreamed of would become possible, is rapidly becoming reality. 
Technologies like Generative AI are rare; they come about once-in-a-lifetime, and completely change what's possible for customers and businesses. 
So, we are investing quite expansively, and, the progress we are making is evident.\n\n
You can see it in what we're rolling out in Alexa+, our next generation Alexa personal assistant that's meaningfully smarter, more capable, and is the first personal assistant that can take significant actions for customers on top of providing intelligent answers to virtually any question.\n\n
You can see it with our AI shopping assistant that's being used by tens of millions of customers around the world to discover new products and make more informed purchase decisions.\n\n
You can see it in an increasing array of shopping features like "Lens" (very cool to be able to take a picture of an item and have it pull up the shopping result), "Buy for Me" (where I can ask our shopping agent to buy an item on another merchant's website for me), or Recommended Size (where we can predict the right size for you based on prior purchases and how different apparel brands run fit-wise relative to each other).\n\n
You can see it in how we're helping our independent sellers more easily create new product detail pages or get advice on how to be even more effective as a seller in our marketplace. Nearly half a million selling partners are using these services, and the listings they're creating are measurably better.\n\n
You can see it in Advertising where we've built a suite of AI tools that make it easier for brands to plan, onboard, create and optimize campaigns. In Q1 alone, over 50K advertisers used these capabilities.\n\n
And, you can see it in what we're delivering in AWS for builders, whether it's custom silicon (Trainium2) to provide better price-performance on model-training and inference, services that make it much easier to build Foundation Models (SageMaker), or leverage leading frontier models and do GenAI inference at scale (Bedrock), our own frontier model (Nova) to give customers leading intelligence at lower latency and cost, or services to make it much easier for developers to write code (Q and QCLI).\n\n
We're also using Generative AI broadly across our internal operations. In our fulfillment network, we're using AI to improve inventory placement, demand forecasting, and the efficiency of our robots—all of which have improved cost to serve and delivery speed. We've rebuilt our Customer Service Chatbot with GenAI, providing an even better experience than we'd had before. And, we're assembling more intelligent and compelling product detail pages from leveraging GenAI.\n\n
I could go on, but you get the idea.\n\n
While we've made a lot of progress, we're still at the relative beginning. 
There are a few reasons we believe this and want to go even faster.\n\n
First, we have strong conviction that AI agents will change how we all work and live. 
Think of agents as software systems that use AI to perform tasks on behalf of users or other systems. 
Agents let you tell them what you want (often in natural language), and do things like scour the web (and various data sources) and summarize results, engage in deep research, write code, find anomalies, highlight interesting insights, translate language and code into other variants, and automate a lot of tasks that consume our time. 
There will be billions of these agents, across every company and in every imaginable field. There will also be agents that routinely do things for you outside of work, from shopping to travel to daily chores and tasks. 
Many of these agents have yet to be built, but make no mistake, they're coming, and coming fast.\n\n
Second, and what makes this agentic future so compelling for Amazon, is that these agents are going to change the scope and speed at which we can innovate for customers. 
Agents will allow us to start almost everything from a more advanced starting point. We'll be able to focus less on rote work and more on thinking strategically about how to improve customer experiences and invent new ones. 
Agents will be teammates that we can call on at various stages of our work, and that will get wiser and more helpful with more experience. 
If we build and leverage the right agents, it's going to rapidly accelerate our ability to make customers lives easier and better every day, and it's going to make our jobs even more exciting and fun than they are today.\n\n
And third, we're going to keep pushing to operate like the world's largest start-up-- customer-obsessed, inventive, fast-moving, lean, scrappy, and full of missionaries trying to build something better for customers and a business that outlasts us all. 
You will continue to see steam and I take actions to help us move faster, have more ownership, and invent more easily. AI will be a substantial catalyst here.\n\n
Today, we have over 1,000 Generative AI services and applications in progress or built, but at our scale, that's a small fraction of what we will ultimately build. 
We're going to lean in further in the coming months. We're going to make it much easier to build agents, and then build (or partner) on several new agents across all of our business units and G&A areas.\n\n
As we roll out more Generative AI and agents, it should change the way our work is done. We will need fewer people doing some of the jobs that are being done today, and more people doing other types of jobs. 
It's hard to know exactly where this nets out over time, but in the next few years, we expect that this will reduce our total corporate workforce as we get efficiency gains from using AI extensively across the company.\n\n
As we go through this transformation together, be curious about AI, educate yourself, attend workshops and take trainings, use and experiment with AI whenever you can, participate in your team's brainstorms to figure out how to invent for our customers more quickly and expansively, and how to get more done with scrappier teams. 
When I first started at Amazon in 1997 as an Assistant Product Manager, I worked on leaner teams that got a lot done quickly and where I could have substantial impact. 
We didn't have tools resembling anything like Generative AI, but we had broad remits, high ambition, and saw the opportunity to improve (and invent) so many customer experiences. 
Fast forward 28 years and the most transformative technology since the Internet is here. 
Those who embrace this change, become conversant in AI, help us build and improve our AI capabilities internally and deliver for customers, will be well-positioned to have high impact and help us reinvent the company.\n\n
There's so much more to come with Generative AI. I'm energized by our progress, excited about our plans ahead, and looking forward to partnering with you all as we change what's possible for our customers, partners, and how we work.\n\n
Andy\n\n
Related news & stories\n\n
More Amazon News\n\n1 / 2\n
</markdown_document>
"""

prediction_content = """
<topic_or_question>\n
Summarize the latest AI news and developments.\n
</topic_or_question>\n\n
<markdown_document>\n
# Update from Amazon CEO Andy Jassy on Generative AI\n\n
# Message from CEO Andy Jassy: Some thoughts on Generative AI\n\n
Written by Andy Jassy, CEO of Amazon\n\nJune 17, 2025\n\n5 min read\n\n
The message below was shared with Amazon employees earlier today.\n\n
Today, in virtually every corner of the company, we're using Generative AI to make customers lives better and easier. 
What started as deep conviction that every customer experience would be reinvented using AI, and that altogether new experiences we've only dreamed of would become possible, is rapidly becoming reality. 
Technologies like Generative AI are rare; they come about once-in-a-lifetime, and completely change what's possible for customers and businesses. 
So, we are investing quite expansively, and, the progress we are making is evident.\n\n
You can see it in what we're rolling out in Alexa+, our next generation Alexa personal assistant that's meaningfully smarter, more capable, and is the first personal assistant that can take significant actions for customers on top of providing intelligent answers to virtually any question.\n\n
You can see it with our AI shopping assistant that's being used by tens of millions of customers around the world to discover new products and make more informed purchase decisions.\n\n
You can see it in an increasing array of shopping features like "Lens" (very cool to be able to take a picture of an item and have it pull up the shopping result), "Buy for Me" (where I can ask our shopping agent to buy an item on another merchant's website for me), or Recommended Size (where we can predict the right size for you based on prior purchases and how different apparel brands run fit-wise relative to each other).\n\n
You can see it in how we're helping our independent sellers more easily create new product detail pages or get advice on how to be even more effective as a seller in our marketplace. Nearly half a million selling partners are using these services, and the listings they're creating are measurably better.\n\n
You can see it in Advertising where we've built a suite of AI tools that make it easier for brands to plan, onboard, create and optimize campaigns. 
In Q1 alone, over 50K advertisers used these capabilities.\n\n
And, you can see it in what we're delivering in AWS for builders, whether it's custom silicon (Trainium2) to provide better price-performance on model-training and inference, services that make it much easier to build Foundation Models (SageMaker), or leverage leading frontier models and do GenAI inference at scale (Bedrock), our own frontier model (Nova) to give customers leading intelligence at lower latency and cost, or services to make it much easier for developers to write code (Q and QCLI).\n\n
We're also using Generative AI broadly across our internal operations. In our fulfillment network, we're using AI to improve inventory placement, demand forecasting, and the efficiency of our robots—all of which have improved cost to serve and delivery speed. 
We've rebuilt our Customer Service Chatbot with GenAI, providing an even better experience than we'd had before. And, we're assembling more intelligent and compelling product detail pages from leveraging GenAI.\n\n
I could go on, but you get the idea.\n\n
While we've made a lot of progress, we're still at the relative beginning. There are a few reasons we believe this and want to go even faster.\n\n
First, we have strong conviction that AI agents will change how we all work and live. 
Think of agents as software systems that use AI to perform tasks on behalf of users or other systems. 
Agents let you tell them what you want (often in natural language), and do things like scour the web (and various data sources) and summarize results, engage in deep research, write code, find anomalies, highlight interesting insights, translate language and code into other variants, and automate a lot of tasks that consume our time. 
There will be billions of these agents, across every company and in every imaginable field. 
There will also be agents that routinely do things for you outside of work, from shopping to travel to daily chores and tasks. 
Many of these agents have yet to be built, but make no mistake, they're coming, and coming fast.\n\n
Second, and what makes this agentic future so compelling for Amazon, is that these agents are going to change the scope and speed at which we can innovate for customers. 
Agents will allow us to start almost everything from a more advanced starting point. We'll be able to focus less on rote work and more on thinking strategically about how to improve customer experiences and invent new ones. 
Agents will be teammates that we can call on at various stages of our work, and that will get wiser and more helpful with more experience. 
If we build and leverage the right agents, it's going to rapidly accelerate our ability to make customers lives easier and better every day, and it's going to make our jobs even more exciting and fun than they are today.\n\n
And third, we're going to keep pushing to operate like the world's largest start-up-- customer-obsessed, inventive, fast-moving, lean, scrappy, and full of missionaries trying to build something better for customers and a business that outlasts us all. 
You will continue to see steam and I take actions to help us move faster, have more ownership, and invent more easily. AI will be a substantial catalyst here.\n\n
Today, we have over 1,000 Generative AI services and applications in progress or built, but at our scale, that's a small fraction of what we will ultimately build. 
We're going to lean in further in the coming months. We're going to make it much easier to build agents, and then build (or partner) on several new agents across all of our business units and G&A areas.\n\n
As we roll out more Generative AI and agents, it should change the way our work is done. We will need fewer people doing some of the jobs that are being done today, and more people doing other types of jobs. 
It's hard to know exactly where this nets out over time, but in the next few years, we expect that this will reduce our total corporate workforce as we get efficiency gains from using AI extensively across the company.\n\n
As we go through this transformation together, be curious about AI, educate yourself, attend workshops and take trainings, use and experiment with AI whenever you can, participate in your team's brainstorms to figure out how to invent for our customers more quickly and expansively, and how to get more done with scrappier teams. 
When I first started at Amazon in 1997 as an Assistant Product Manager, I worked on leaner teams that got a lot done quickly and where I could have substantial impact. 
We didn't have tools resembling anything like Generative AI, but we had broad remits, high ambition, and saw the opportunity to improve (and invent) so many customer experiences. 
Fast forward 28 years and the most transformative technology since the Internet is here. 
Those who embrace this change, become conversant in AI, help us build and improve our AI capabilities internally and deliver for customers, will be well-positioned to have high impact and help us reinvent the company.\n\n
There's so much more to come with Generative AI. I'm energized by our progress, excited about our plans ahead, and looking forward to partnering with you all as we change what's possible for our customers, partners, and how we work.\n\n
Andy\n\nRelated news & stories\n\nMore Amazon News\n\n1 / 2\n
</markdown_document>
"""

models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
results_with_prediction = {}
results_without_prediction = {}

print("Testing latency for different GPT-4.1 models (WITH and WITHOUT prediction)...\n")

# Test with prediction
print("=" * 50)
print("TESTING WITH PREDICTION")
print("=" * 50)

for model in models:
    print(f"Testing {model} with prediction...")
    
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            prediction={"type": "content", "content": prediction_content},
        )
        end_time = time.time()
        
        latency = end_time - start_time
        results_with_prediction[model] = latency
        print(f"  ✓ {model}: {latency:.3f} seconds")
        
        # Print prediction token details
        if hasattr(completion, 'usage') and completion.usage and hasattr(completion.usage, 'completion_tokens_details'):
            details = completion.usage.completion_tokens_details
            print(f"    Prediction tokens - Accepted: {details.accepted_prediction_tokens}, Rejected: {details.rejected_prediction_tokens}")
        
    except Exception as e:
        print(f"  ✗ {model}: Error - {str(e)}")
        results_with_prediction[model] = None
    
    print()

# Test without prediction
print("=" * 50)
print("TESTING WITHOUT PREDICTION")
print("=" * 50)

for model in models:
    print(f"Testing {model} without prediction...")
    
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        end_time = time.time()
        
        latency = end_time - start_time
        results_without_prediction[model] = latency
        print(f"  ✓ {model}: {latency:.3f} seconds")
        
    except Exception as e:
        print(f"  ✗ {model}: Error - {str(e)}")
        results_without_prediction[model] = None
    
    print()

# Display results
print("=" * 60)
print("LATENCY COMPARISON RESULTS")
print("=" * 60)

print("\nWITH PREDICTION:")
print("-" * 30)
valid_with = {k: v for k, v in results_with_prediction.items() if v is not None}
if valid_with:
    for model in models:
        if model in valid_with:
            print(f"{model:<15}: {valid_with[model]:.3f} seconds")

print("\nWITHOUT PREDICTION:")
print("-" * 30)
valid_without = {k: v for k, v in results_without_prediction.items() if v is not None}
if valid_without:
    for model in models:
        if model in valid_without:
            print(f"{model:<15}: {valid_without[model]:.3f} seconds")

print("\nDIFFERENCE (Without - With):")
print("-" * 30)
for model in models:
    if model in valid_with and model in valid_without:
        diff = valid_without[model] - valid_with[model]
        print(f"{model:<15}: {diff:+.3f} seconds")

if valid_with and valid_without:
    print("\nSUMMARY:")
    print("-" * 30)
    
    fastest_with = min(valid_with, key=valid_with.get)
    fastest_without = min(valid_without, key=valid_without.get)
    
    print(f"Fastest with prediction:    {fastest_with} ({valid_with[fastest_with]:.3f}s)")
    print(f"Fastest without prediction: {fastest_without} ({valid_without[fastest_without]:.3f}s)")