# wrappers/llama/llm_client.py
# LLMClientWrapper for local Llama via an OpenAI-compatible endpoint (e.g., Ollama).
# Matches the same interface used by components: .chat(system_prompt, user_prompt, ...)->str

import os
import time
import json
import random
from datetime import datetime
import traceback
import openai  # uses the official OpenAI SDK; we point it at a local base_url

class LLMClientWrapper:
    def __init__(
        self,
        llm_api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 10,
        log_prompts: bool = True,
        log_dir: str = "logs/llm/",
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        # Read config with sensible defaults for Ollama
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = llm_api_key or os.getenv("LLM_API_KEY", "ollama")   # any non-empty string is fine locally
        self.model = model or os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")

        # Client + behavior
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.max_retries = max_retries
        self.temperature = temperature
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
            {"role": "user",   "content": user_prompt},
        ]

        log_entry = {
            "type": log_type,
            "repo": repo,
            "pr_number": pr_number,
            "provider": "llama-local",
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": system_prompt,
            "user_prompt_preview": user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""),
            "timestamp": datetime.utcnow().isoformat(),
            "retries": [],
        }

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                dt = time.time() - t0
                content = (
                    (resp.choices[0].message.content or "").strip()
                    if resp and getattr(resp, "choices", None)
                    else "[LLM returned no choices]"
                )

                log_entry["duration"] = f"{dt:.2f}s"
                log_entry["response"] = content
                if self.log_prompts:
                    self._write_log(log_entry)

                # light pacing so we don't hammer the local server
                time.sleep(0.2)
                return content

            except Exception as e:
                dt = time.time() - t0
                err = f"{type(e).__name__}: {e}"
                tb  = traceback.format_exc()

                retry = {
                    "attempt": attempt,
                    "error": err,
                    "duration": f"{dt:.2f}s",
                    "traceback": tb,
                }

                # Simple backoff; most local errors are transient (model cold start, etc.)
                wait = min((2 ** attempt) + random.random(), 30)
                retry["retry_after"] = f"{wait:.2f}s"
                log_entry["retries"].append(retry)

                if attempt == self.max_retries:
                    log_entry["error"] = err
                    log_entry["traceback"] = tb
                    if self.log_prompts:
                        self._write_log(log_entry)
                    return f"[Local LLM error] {err}"

                time.sleep(wait)

        log_entry["error"] = "Max retries exceeded"
        if self.log_prompts:
            self._write_log(log_entry)
        return "[LLM failed after maximum retries]"

    def _write_log(self, entry: dict):
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        repo_slug = entry.get("repo", "unknown").replace("/", "_")
        log_type = entry.get("type", "general")
        prn = entry.get("pr_number", 0)
        fname = f"llama_{log_type}_{repo_slug}_pr{prn}_{ts}.json"
        try:
            with open(os.path.join(self.log_dir, fname), "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
        except Exception as e:
            print(f"[Warn] failed to write log: {e}")
