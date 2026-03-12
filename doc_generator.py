import anthropic
import os
import json

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")
client = anthropic.Anthropic(api_key=api_key)

def generate_docs(code: str) -> str:
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # More stable model
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""You are an expert API documentation writer. Analyze this code and return clean, professional markdown documentation only. No explanations.

Code:
{code}"""
            }]
        )
        return message.content[0].text
    except Exception as e:
        raise ValueError(f"Docs generation failed: {str(e)}")

def generate_contract(code: str) -> dict:
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""You are an OpenAPI contract generator.
Analyze this API code and generate a valid OpenAPI 3.0 specification as JSON only.
Return ONLY valid JSON. No explanation. No markdown. No backticks.

Code:
{code}"""
            }]
        )
        text = message.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Contract generation failed: {str(e)}")

