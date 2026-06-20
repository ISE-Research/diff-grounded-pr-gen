# wrappers/mistral/llm_client.py
# LLMClientWrapper provides a robust, retry-safe interface to interact with the Mistral LLM API.
# Supports logging, exponential backoff, prompt-response auditing, and contextual error tracing.

import time
import random
import os
import json
from datetime import datetime
from mistralai import Mistral


class LLMClientWrapper:
    def __init__(
        self,
        llm_api_key: str,
        model: str = "mistral-large-latest",
        max_retries: int = 10,
        log_prompts: bool = True,
        log_dir: str = "logs/llm/",
        temperature: float = 0.2,
    ):
        # Initialize Mistral client and logging configuration
        self.client = Mistral(api_key=llm_api_key)
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.log_prompts = log_prompts
        self.log_dir = os.path.normpath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    # Sends a chat message to Mistral with logging and retry handling
    def chat(self, system_prompt: str, user_prompt: str, log_type: str = "general", repo: str = "unknown", pr_number: int = 0) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
            "retries": []
        }

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                duration = time.time() - start_time

                content = (
                    response.choices[0].message.content.strip()
                    if response.choices else "[LLM returned no choices]"
                )

                log_entry["duration"] = f"{duration:.2f}s"
                log_entry["response"] = content

                if self.log_prompts:
                    self._write_log(log_entry)

                print(
                    f"[LLMClientWrapper] Success in {duration:.2f}s — "
                    f"Type: {log_type} | Repo: {repo} | PR #{pr_number}\n"
                    f"Response: {content[:100]}{'...' if len(content) > 100 else ''}\n"
                )

                time.sleep(5)

                return content

            except Exception as e:
                error_msg = str(e)
                duration = time.time() - start_time

                retry_info = {
                    "attempt": attempt,
                    "error": error_msg,
                    "duration": f"{duration:.2f}s"
                }

                if "429" in error_msg or "rate" in error_msg.lower() or "capacity" in error_msg.lower():
                    wait_time = min((2 ** attempt) + random.uniform(0, 1.5), 180)
                    retry_info["retry_after"] = f"{wait_time:.2f}s"
                    log_entry["retries"].append(retry_info)

                    print(f"[Retry {attempt}] Rate limit hit. Waiting {wait_time:.2f}s...\n")
                    time.sleep(wait_time)
                else:
                    retry_info["retry_after"] = "none"
                    log_entry["retries"].append(retry_info)
                    log_entry["error"] = error_msg

                    if self.log_prompts:
                        self._write_log(log_entry)

                    return f"[Error from Mistral: {error_msg}]"

        log_entry["error"] = "Max retries exceeded"
        if self.log_prompts:
            self._write_log(log_entry)

        return "[LLM failed after maximum retries]"

    # Writes logs with filename including log_type, repo, pr_number, and timestamp
    def _write_log(self, log_entry: dict):
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        repo_slug = log_entry.get("repo", "unknown").replace("/", "_")
        log_type = log_entry.get("type", "general")
        pr_number = log_entry.get("pr_number", 0)

        filename = f"mistral_{log_type}_{repo_slug}_pr{pr_number}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to write LLM log: {e}\n")
