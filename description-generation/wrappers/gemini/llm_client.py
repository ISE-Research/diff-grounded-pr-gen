# wrappers/gemini/llm_client.py
# LLMClientWrapper for Gemini via Google GenAI SDK.

import os
import time
import json
import random
import traceback
from datetime import datetime

from google import genai


class LLMClientWrapper:
    def __init__(
        self,
        llm_api_key: str,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 10,
        log_prompts: bool = True,
        log_dir: str = "logs/llm/",
        temperature: float = 0.2,
    ):
        self.api_key = llm_api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.max_retries = max_retries
        self.temperature = temperature
        self.log_prompts = log_prompts
        self.log_dir = os.path.normpath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.client = genai.Client(api_key=self.api_key)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        log_type: str = "general",
        repo: str = "unknown",
        pr_number: int = 0,
    ) -> str:
        log_entry = {
            "type": log_type,
            "repo": repo,
            "pr_number": pr_number,
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": system_prompt,
            "user_prompt_preview": user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""),
            "timestamp": datetime.utcnow().isoformat(),
            "retries": [],
        }

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                prompt_text = f"{system_prompt}\n\n{user_prompt}".strip()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    generation_config={"temperature": self.temperature},
                )
                duration = time.time() - start_time

                content = (getattr(response, "text", "") or "").strip()
                if not content:
                    content = "[LLM returned empty response]"

                log_entry["duration"] = f"{duration:.2f}s"
                log_entry["response"] = content

                if self.log_prompts:
                    self._write_log(log_entry)

                time.sleep(1)
                return content

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"{type(e).__name__}: {e}"
                traceback_str = traceback.format_exc()

                retry_info = {
                    "attempt": attempt,
                    "error": error_msg,
                    "duration": f"{duration:.2f}s",
                    "traceback": traceback_str,
                }

                wait_time = min((2 ** attempt) + random.uniform(0, 1.5), 60)
                retry_info["retry_after"] = f"{wait_time:.2f}s"
                log_entry["retries"].append(retry_info)

                if attempt == self.max_retries:
                    log_entry["error"] = error_msg
                    log_entry["traceback"] = traceback_str
                    if self.log_prompts:
                        self._write_log(log_entry)
                    return f"[Error from Gemini: {error_msg}]"

                time.sleep(wait_time)

        log_entry["error"] = "Max retries exceeded"
        if self.log_prompts:
            self._write_log(log_entry)
        return "[LLM failed after maximum retries]"

    def _write_log(self, entry: dict):
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        repo_slug = entry.get("repo", "unknown").replace("/", "_")
        log_type = entry.get("type", "general")
        prn = entry.get("pr_number", 0)
        fname = f"gemini_{log_type}_{repo_slug}_pr{prn}_{ts}.json"
        try:
            with open(os.path.join(self.log_dir, fname), "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to write LLM log: {e}\n")
