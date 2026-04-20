import asyncio
import os

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from evals.processing import utils

_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0


async def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    if model.startswith("gemini"):
        if not os.getenv("GOOGLE_GEMINI_API_KEY"):
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable is not set.")
        fn = _call_gemini
    elif model.startswith("gpt"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        fn = _call_openai
    else:
        raise ValueError(
            f"Unsupported model: {model}, please use Gemini or OpenAI models only."
        )

    for attempt in range(_MAX_RETRIES):
        try:
            return await fn(model, system_prompt, user_prompt)
        except Exception as e:
            print(f"Ran into error when calling {model}: {e}")
            if attempt == _MAX_RETRIES - 1:
                raise e
            print(f"Retrying in {_BASE_BACKOFF ** (attempt + 1)} seconds...")
            await asyncio.sleep(_BASE_BACKOFF ** (attempt + 1))


async def _call_gemini(model: str, system_prompt: str, user_prompt: str) -> str:
    client = genai.Client(
        http_options=HttpOptions(api_version="v1beta"),
        api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        safety_settings=[
            types.SafetySetting(category=category, threshold="BLOCK_NONE")
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_CIVIC_INTEGRITY",
            ]
        ],
    )
    response = await client.aio.models.generate_content(
        model=model, contents=user_prompt, config=config
    )
    return response.text


async def _call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    result = await utils.post_json(
        "https://api.openai.com/v1/chat/completions", payload, headers
    )
    return result["choices"][0]["message"]["content"]
