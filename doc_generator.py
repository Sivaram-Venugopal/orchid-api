import anthropic
import os
import json

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def generate_docs(code: str) -> str:
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""You are an expert API documentation writer.
         
Return clean markdown only."""
        }]
    )
    return message.content[0].text

def generate_contract(code: str) -> dict:
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""You are an OpenAPI contract generator.
Analyze this API code and generate a valid OpenAPI 3.0 specification.
Return ONLY valid JSON. No explanation. No markdown. No backticks.

Code:
{code}"""
        }]
    )
    text = message.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

