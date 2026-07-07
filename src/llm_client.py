"""
Groq API client wrapper.
Handles both the classification call and the LLM-as-judge scoring call.
Free tier: https://console.groq.com — no credit card required.
"""
import os
import json
import time
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_client = None


def get_client() -> AsyncGroq:
    """
    Lazily creates the Groq client on first actual use. This lets the rest
    of the codebase (schemas, scorer, comparator, tests) be imported and
    unit-tested without requiring an API key to be set.
    """
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and add it to your .env file."
            )
        _client = AsyncGroq(api_key=GROQ_API_KEY)
    return _client


async def call_groq(
    system_prompt: str,
    user_message: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 300,
    max_retries: int = 3,
) -> dict:
    """
    Calls the Groq chat completion endpoint and returns parsed JSON output
    plus timing/token metadata. Retries on transient errors / rate limits.
    """
    last_error = None

    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            response = await get_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - start) * 1000

            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)

            return {
                "parsed": parsed,
                "raw_text": raw_text,
                "latency_ms": latency_ms,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "error": None,
            }

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
        except Exception as e:
            last_error = str(e)
            # Rate limit backoff
            if "rate_limit" in str(e).lower() or "429" in str(e):
                await asyncio.sleep(2 ** attempt)
                continue

        await asyncio.sleep(1)

    # All retries exhausted
    return {
        "parsed": None,
        "raw_text": None,
        "latency_ms": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": last_error,
    }


JUDGE_SYSTEM_PROMPT = """You are grading how well a generated summary matches an
expected/reference summary of a customer support email. Score from 1 to 5:

5 = captures the same core issue and intent, wording can differ
4 = captures the core issue, missing a minor nuance
3 = partially correct, misses something important
2 = mostly wrong but tangentially related
1 = completely wrong or irrelevant

Respond ONLY with valid JSON: {"score": <int 1-5>, "reason": "<short reason>"}
"""


async def judge_summary(expected_summary: str, actual_summary: str, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Uses a smaller/faster Groq model as an LLM judge to score summary quality.
    Using a cheaper model here keeps the eval suite fast and free-tier friendly.
    """
    user_message = (
        f"Expected summary: {expected_summary}\n"
        f"Actual summary: {actual_summary}\n\n"
        "Score the actual summary against the expected one."
    )
    result = await call_groq(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        temperature=0.0,
        max_tokens=100,
    )
    if result["error"] or not result["parsed"]:
        return {"score": 1, "reason": f"Judge call failed: {result['error']}"}
    return result["parsed"]
