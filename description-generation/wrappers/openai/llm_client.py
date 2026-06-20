# wrappers/openai/llm_client.py
import time
import random
import os
import json
import traceback
from datetime import datetime
import openai


class LLMClientWrapper:
    def __init__(
        self,
        llm_api_key: str,
        model: str = "gpt-5-mini-2025-08-07",
        max_retries: int = 10,
        log_prompts: bool = True,
        log_dir: str = "logs/llm/",
        temperature: float | None = None,
    ):
        # Setup OpenAI client and config
        self.client = openai.OpenAI(api_key=llm_api_key)
        self.model = model
        self.max_retries = max_retries
        self.temperature = 1 if temperature is None else temperature
        self.log_prompts = log_prompts
        self.log_dir = os.path.normpath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        log_type: str = "general",
        repo: str = "unknown",
        pr_number: int = 0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        log_entry = {
            "type": log_type,
            "repo": repo,
            "pr_number": pr_number,
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": system_prompt,
            "user_prompt_preview": user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""),
            "timestamp": datetime.utcnow().isoformat(),
            "retries": [],
        }

        def _is_context_error(message: str) -> bool:
            msg = (message or "").lower()
            return any(
                phrase in msg
                for phrase in (
                    "request too large",
                    "tokens per min",
                    "tpm",
                    "context length",
                    "maximum context",
                    "prompt is too long",
                    "input or output tokens must be reduced",
                    "token limit",
                )
            )

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                # Call OpenAI chat completion
                def _call_with_temperature(include_temperature: bool):
                    temp_kwargs = {}
                    if include_temperature and self.temperature is not None:
                        temp_kwargs["temperature"] = self.temperature
                    return self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **temp_kwargs,
                    )

                try:
                    response = _call_with_temperature(include_temperature=True)
                except Exception as inner_exc:
                    msg = str(inner_exc)
                    if "temperature" in msg and "unsupported" in msg.lower():
                        response = _call_with_temperature(include_temperature=False)
                    else:
                        raise

                duration = time.time() - start_time

                # Handle new SDK return type
                if response.choices:
                    msg = response.choices[0].message
                    content = msg.content.strip()
                else:
                    content = "[LLM returned no choices]"

                log_entry["duration"] = f"{duration:.2f}s"
                log_entry["response"] = content

                if self.log_prompts:
                    self._write_log(log_entry)

                print(
                    f"[LLMClientWrapper] Success in {duration:.2f}s — "
                    f"Type: {log_type} | Repo: {repo} | PR #{pr_number}\n"
                    f"Response: {content[:100]}{'...' if len(content) > 100 else ''}\n"
                )

                return content

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                duration = time.time() - start_time

                retry_info = {
                    "attempt": attempt,
                    "error": error_msg,
                    "traceback": traceback_str,
                    "duration": f"{duration:.2f}s",
                }

                is_rate = "rate" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower()
                is_context = _is_context_error(error_msg) or _is_context_error(traceback_str)
                if is_rate:
                    if is_context and attempt > 2:
                        log_entry["retries"].append(retry_info)
                        log_entry["error"] = error_msg
                        log_entry["traceback"] = traceback_str
                        if self.log_prompts:
                            self._write_log(log_entry)
                        return "[LLM skipped: context_too_large]"
                    wait_time = min((2 ** attempt) + random.uniform(0, 1.5), 180)
                    retry_info["retry_after"] = f"{wait_time:.2f}s"
                    log_entry["retries"].append(retry_info)

                    print(f"[Retry {attempt}] Rate limit hit. Waiting {wait_time:.2f}s...\n")
                    print(f"[Detailed Error]: {traceback_str}")
                    time.sleep(wait_time)
                else:
                    retry_info["retry_after"] = "none"
                    log_entry["retries"].append(retry_info)
                    log_entry["error"] = error_msg
                    log_entry["traceback"] = traceback_str

                    print(f"[LLMClientWrapper Error] {error_msg}\n{traceback_str}")

                    if self.log_prompts:
                        self._write_log(log_entry)

                    return f"[Error from OpenAI: {error_msg}]"

        log_entry["error"] = "Max retries exceeded"
        if self.log_prompts:
            self._write_log(log_entry)

        return "[LLM failed after maximum retries]"

    def _write_log(self, log_entry: dict):
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        repo_slug = log_entry.get("repo", "unknown").replace("/", "_")
        log_type = log_entry.get("type", "general")
        pr_number = log_entry.get("pr_number", 0)

        filename = f"openai_{log_type}_{repo_slug}_pr{pr_number}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to write LLM log: {e}\n")
