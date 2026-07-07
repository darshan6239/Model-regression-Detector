"""
Pydantic schemas for prompt configuration and classifier I/O.
This is the "interface contract" the whole eval pipeline is built against.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class FewShotExample(BaseModel):
    input: str
    output: dict


class PromptConfig(BaseModel):
    """Represents one versioned prompt. Loaded from a YAML file in /prompts."""
    version: str
    created_at: str
    model: str = "llama-3.3-70b-versatile"
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 300


class ClassificationInput(BaseModel):
    email_text: str


class ClassificationOutput(BaseModel):
    """Structured output the classifier must return."""
    category: Literal["billing", "technical", "account", "general"]
    summary: str


class TestCase(BaseModel):
    """One entry in the golden dataset."""
    id: str
    input: ClassificationInput
    expected: ClassificationOutput
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    notes: Optional[str] = None


class TestResult(BaseModel):
    """Result of running one test case through the classifier."""
    test_case_id: str
    input_text: str
    expected_category: str
    actual_category: str
    category_match: bool
    expected_summary: str
    actual_summary: str
    summary_score: int  # 1-5, from LLM-as-judge
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    difficulty: str
    error: Optional[str] = None


class RunMetadata(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    timestamp: str
    dataset_version: str
