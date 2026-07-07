"""
The actual LLM feature under test: a customer support email classifier.
This is deliberately simple — the point of this project is the eval
pipeline around it, not the feature itself.
"""
import yaml
from pathlib import Path
from src.llm_client import call_groq

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt_config(version: str) -> dict:
    """Loads a versioned prompt YAML file, e.g. load_prompt_config('v1')."""
    path = PROMPTS_DIR / f"{version}_classifier.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file found for version '{version}' at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


async def classify_email(email_text: str, prompt_config: dict) -> dict:
    """
    Runs one email through the classifier using the given prompt config.
    Returns the raw call_groq result (parsed output + timing + tokens).
    """
    system_prompt = prompt_config["system_prompt"]

    few_shot = prompt_config.get("few_shot_examples", [])
    if few_shot:
        examples_text = "\n\nExamples:\n"
        for ex in few_shot:
            examples_text += f"\nEmail: {ex['input']}\nOutput: {ex['output']}\n"
        system_prompt = system_prompt + examples_text

    result = await call_groq(
        system_prompt=system_prompt,
        user_message=f"Email:\n{email_text}",
        model=prompt_config.get("model", "llama-3.3-70b-versatile"),
        temperature=prompt_config.get("temperature", 0.0),
        max_tokens=prompt_config.get("max_tokens", 300),
    )
    return result
