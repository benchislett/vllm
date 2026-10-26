# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EvalScope GSM8K protocol for server and offline engine tests."""


def gsm8k_task(config: dict) -> dict:
    """Translate model-test settings into the standard four-shot chat evaluation."""
    return {
        "model": config["model_name"],
        "datasets": ["gsm8k"],
        "dataset_hub": "huggingface",
        "limit": config.get("num_questions", 1319),
        "seed": config.get("seed", 42),
        "eval_batch_size": config.get("max_concurrency") or 32,
        "generation_config": {
            "timeout": config.get("request_timeout_seconds", 600),
            "max_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.0),
            "extra_body": {
                "chat_template_kwargs": config.get(
                    "chat_template_kwargs", {"enable_thinking": False}
                ),
            },
        },
    }


def load_gsm8k(num_questions: int):
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    task = TaskConfig(
        **gsm8k_task({"model_name": "offline", "num_questions": num_questions}),
        eval_type="openai_api",
    )
    adapter = get_benchmark("gsm8k", task)
    samples = list(adapter.load_dataset()["main"])
    return adapter, samples


def gsm8k_prompts(num_questions: int) -> list[list[dict]]:
    if num_questions == 0:
        return []
    _, samples = load_gsm8k(num_questions)
    return [
        [message.model_dump(include={"role", "content"}) for message in sample.input]
        for sample in samples
    ]


def evaluate_gsm8k_offline(
    llm,
    num_questions: int = 1319,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    chat_template_kwargs: dict[str, object] | None = None,
) -> dict[str, float | int]:
    """Use the same EvalScope prompts/scorer, with LLM.chat applying the template."""
    from evalscope.api.evaluator import TaskState
    from evalscope.api.model import ModelOutput
    from evalscope.constants import ScoreStatus

    from vllm import SamplingParams

    adapter, samples = load_gsm8k(num_questions)
    conversations = [
        [message.model_dump(include={"role", "content"}) for message in sample.input]
        for sample in samples
    ]
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=temperature, max_tokens=max_tokens),
        chat_template_kwargs=(
            chat_template_kwargs
            if chat_template_kwargs is not None
            else {"enable_thinking": False}
        ),
    )
    scores = []
    for sample, output in zip(samples, outputs, strict=True):
        state = TaskState(
            model="offline",
            sample=sample,
            completed=True,
            output=ModelOutput.from_content("offline", output.outputs[0].text),
        )
        score = adapter.calculate_metrics(state).score
        if score.status != ScoreStatus.SUCCESS or score.main_value is None:
            raise RuntimeError(f"GSM8K scoring failed: {score.metadata}")
        scores.append(float(score.main_value))
    if not scores:
        raise ValueError("GSM8K evaluation produced no scored samples")
    return {"accuracy": sum(scores) / len(scores), "num_questions": len(scores)}


if __name__ == "__main__":
    raise SystemExit(
        "Use 'vllm eval gsm8k' for the standard chat-templated evaluation."
    )
