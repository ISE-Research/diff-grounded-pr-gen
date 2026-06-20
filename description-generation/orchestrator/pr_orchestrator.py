# orchestrator.py
# Coordinates all components for automatic pull request description generation.

from typing import Any, Dict
from pathlib import Path

from wrappers.github_wrapper import GitHubWrapper
from components.pr_data_collector import PullRequestDataCollector
from components.pr_description_generator import PRDescriptionGenerator

# CMG-related imports
from components.cmg_commit_rewriter import CommitMessageRewriter
from components.ranking import compute_file_scores, rank_commits


def _resolve_llm_client(provider: str):
    provider = (provider or "openai").lower()
    if provider == "llama":
        from wrappers.llama.llm_client import LLMClientWrapper as Client
    elif provider == "openai":
        from wrappers.openai.llm_client import LLMClientWrapper as Client
    elif provider == "mistral":
        from wrappers.mistral.llm_client import LLMClientWrapper as Client
    elif provider == "deepseek":
        from wrappers.deepseek.llm_client import LLMClientWrapper as Client
    elif provider == "gemini":
        from wrappers.gemini.llm_client import LLMClientWrapper as Client
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'")
    return Client


class PRDescriptionOrchestrator:
    # def __init__(self, github_token: str, mistral_api_key: str):
    def __init__(
        self,
        github_token: str | None,
        llm_api_key: str | None = None,
        enable_llm_components: bool = True,
        enable_data_collection: bool = True,
        llm_provider: str = "openai",
        llm_settings: Dict[str, Any] | None = None,
        cmg_config: Dict[str, Any] | None = None,
        ranking_config: Dict[str, Any] | None = None,
    ):
        print("[ORCHESTRATOR] Initializing wrappers and components...\n")

        # Wrappers for GitHub API and LLM
        self.github_wrapper: GitHubWrapper | None = None
        self.data_collector: PullRequestDataCollector | None = None

        # CMG / commit rewriting configuration
        self.cmg_config = cmg_config or {}
        self.use_cmg: bool = bool(self.cmg_config.get("enabled", False))
        self.commit_rewriter: CommitMessageRewriter | None = None
        self.llm_provider = (llm_provider or "openai").lower()
        self.llm_settings = llm_settings or {}
        self.ranking_config = ranking_config or {}

        if enable_data_collection:
            if not github_token:
                raise ValueError("GitHub token is required when data collection is enabled.")
            self.github_wrapper = GitHubWrapper(github_token)
            self.data_collector = PullRequestDataCollector(self.github_wrapper)

        # Core components for PR processing
        self.llm_client: LLMClientWrapper | None = None
        self.description_generator: PRDescriptionGenerator | None = None

        if enable_llm_components:
            llm_client_cls = _resolve_llm_client(self.llm_provider)
            client_kwargs: Dict[str, Any] = {}
            model_name = self.llm_settings.get("model")
            if model_name:
                client_kwargs["model"] = model_name
            if "log_prompts" in self.llm_settings:
                client_kwargs["log_prompts"] = self.llm_settings["log_prompts"]
            if "temperature" in self.llm_settings:
                client_kwargs["temperature"] = self.llm_settings["temperature"]
            if self.llm_provider == "llama":
                if "base_url" in self.llm_settings:
                    client_kwargs["base_url"] = self.llm_settings["base_url"]
            log_dir = self.llm_settings.get("log_dir")
            if not log_dir:
                root_dir = Path(__file__).resolve().parents[2]
                log_dir = root_dir / "logs" / "pr-descriptions" / "llm"
            client_kwargs["log_dir"] = str(log_dir)
            self.llm_client = llm_client_cls(llm_api_key, **client_kwargs)

            # Existing single-call PR description generator (main-branch behavior)
            self.description_generator = PRDescriptionGenerator(
                self.llm_client,
                ranking_config=self.ranking_config,
            )

            # Initialize CMG commit rewriter if enabled
            if self.use_cmg:
                print(
                    "[ORCHESTRATOR][CMG] enabled; demo_source=knowledge_graph"
                )

                self.commit_rewriter = CommitMessageRewriter(
                    llm_client=self.llm_client,
                    settings=self.cmg_config,
                )

    def run(self, repo_name: str, pr_number: int) -> Dict[str, Any]:
        """
        End-to-end: collect PR context via GitHub and then generate a description.
        """
        if not self.description_generator:
            raise RuntimeError("LLM components are disabled; cannot generate pull request descriptions.")

        if not self.data_collector:
            raise RuntimeError("Data collection is disabled; provide context manually instead.")

        print(f"[ORCHESTRATOR] Processing PR #{pr_number} in repo '{repo_name}'...\n")

        # Step 1: Collect all relevant PR data
        pr_data = self.data_collector.collect(repo_name, pr_number)
        return self._generate_description_from_context(pr_data, repo_name=repo_name, pr_number=pr_number)

    def collect_pr_context(self, repo_name: str, pr_number: int) -> dict:
        """
        Collect raw PR data without invoking any LLM-powered components.
        Used for building the knowledge graph.
        """
        if not self.data_collector:
            raise RuntimeError("Data collection is disabled; cannot fetch context via API.")

        print(f"[ORCHESTRATOR] Collecting context for PR #{pr_number} in repo '{repo_name}'...\n")
        return self.data_collector.collect(repo_name, pr_number)

    def generate_from_context(
        self,
        pr_context: dict,
        repo_name: str | None = None,
        pr_number: int | None = None,
        generation_options: Dict[str, Any] | None = None,
        precomputed_file_summaries: list[dict] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a PR description and related artifacts given a pr_context
        (e.g., reconstructed from the knowledge graph).
        """
        if not self.description_generator:
            raise RuntimeError("LLM components are disabled; cannot generate pull request descriptions.")

        repo = repo_name or pr_context.get("repo")
        pr_num = pr_number or pr_context.get("pr_number")
        if repo is None or pr_num is None:
            raise ValueError("Repository name and PR number must be provided in context or explicitly.")

        return self._generate_description_from_context(
            pr_context,
            repo_name=repo,
            pr_number=pr_num,
            generation_options=generation_options,
            precomputed_file_summaries=precomputed_file_summaries,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_cmg_on_commits(
        self,
        pr_context: dict,
        repo_name: str,
        pr_number: int,
    ) -> None:
        """
        If CMG is enabled, run the research-grade commit message generation
        pipeline *only* on the commits from pr_context, and write the final
        messages back into pr_context['commits'].

        This uses CommitMessageRewriter.rewrite_if_needed(...) from the CMG
        branch (not rewrite_commits), which:
          - decides if a commit needs CMG
          - returns a list of {sha, message} dicts
        """
        if not self.use_cmg:
            return
        if not self.commit_rewriter:
            print("[ORCHESTRATOR][CMG] Config enabled but commit_rewriter is not initialized.")
            return

        commits = pr_context.get("commits") or []
        if not commits:
            print("[ORCHESTRATOR][CMG] No commits in pr_context; skipping CMG.")
            return

        selected_shas: set[str] | None = None
        commit_cfg = (self.ranking_config.get("commit") or {})
        include_all_if_leq = int(commit_cfg.get("include_all_if_commit_count_leq") or 0)
        top_k = int(commit_cfg.get("top_k_large") or 0)
        if top_k and (include_all_if_leq == 0 or len(commits) > include_all_if_leq):
            file_weights = (self.ranking_config.get("file") or {}).get("weights") or {}
            file_scores = compute_file_scores(pr_context.get("files") or [], file_weights)
            ranked = rank_commits(commits, file_scores, commit_cfg.get("weights") or {})
            selected_shas = {sha for sha, _ in ranked[:top_k]}
            print(f"[ORCHESTRATOR][CMG] Ranked commit selection: {len(selected_shas)}/{len(commits)} commits.")

        print(
            f"[ORCHESTRATOR][CMG] Running CMG on {len(commits)} commits for "
            f"{repo_name}#{pr_number}..."
        )

        # CMG API from pr_descriptions_cmg/components/cmg_commit_rewriter.py:
        # def rewrite_if_needed(self, commits: List[Dict], repo_name: str, pr_number: int) -> List[Dict]:
        rewritten_commits = self.commit_rewriter.rewrite_if_needed(
            commits=commits,
            repo_name=repo_name,
            pr_number=pr_number,
            selected_shas=selected_shas,
        )

        # Map CMG output by SHA for safer merge back into pr_context
        by_sha: Dict[str, Dict[str, Any]] = {}
        for c in rewritten_commits or []:
            sha = c.get("sha")
            if sha:
                by_sha[sha] = c

        updated_count = 0

        for commit in commits:
            sha = commit.get("sha")
            if not sha:
                continue

            updated = by_sha.get(sha)
            if not updated:
                continue

            new_msg = updated.get("message")
            if not new_msg:
                continue

            old_msg = commit.get("message") or ""

            commit["cmg_status"] = updated.get("status")
            commit["cmg_reason"] = updated.get("reason")
            commit["cmg_candidate_message"] = updated.get("candidate_message")
            commit["cmg_final_message"] = updated.get("final_message")
            commit["cmg_patch_available"] = updated.get("patch_available")

            if old_msg and old_msg != new_msg:
                commit["cmg_rewritten_message"] = new_msg
                updated_count += 1
            else:
                commit.pop("cmg_rewritten_message", None)

        print(
            f"[ORCHESTRATOR][CMG] Updated commit messages for {updated_count} commits "
            f"out of {len(commits)} total."
        )
        pr_context["_cmg_done"] = True

    def _generate_description_from_context(
        self,
        pr_context: dict,
        repo_name: str,
        pr_number: int,
        generation_options: Dict[str, Any] | None = None,
        precomputed_file_summaries: list[dict] | None = None,
    ) -> Dict[str, Any]:
        """
        Main internal entrypoint:
        - (Optionally) run CMG to upgrade commit messages in pr_context.
        - Call the existing PRDescriptionGenerator to produce:
          description, rewritten_commits, file_summaries, raw_response.
        """
        generation_options = generation_options or {}
        use_cmg = bool(generation_options.get("use_cmg", True))
        include_file_summaries = bool(generation_options.get("include_file_summaries", True))
        include_commits = bool(generation_options.get("include_commits", True))
        effective_use_cmg = use_cmg and self.use_cmg

        # NEW: run CMG on commit messages (if enabled) before generating the PR description
        if effective_use_cmg and pr_context.get("_cmg_done"):
            print("[ORCHESTRATOR][CMG] Reusing cached CMG rewrites for this PR.")
        elif effective_use_cmg:
            self._run_cmg_on_commits(pr_context, repo_name=repo_name, pr_number=pr_number)
        else:
            print("[ORCHESTRATOR][CMG] Skipping CMG for this generation mode.")

        # Existing single-call generator (now sees CMG-upgraded commit messages)
        outputs = self.description_generator.generate_outputs(
            pr_context=pr_context,
            repo_name=repo_name,
            pr_number=pr_number,
            include_file_summaries=include_file_summaries,
            include_commits=include_commits,
            use_cmg_commits=effective_use_cmg,
            precomputed_file_summaries=precomputed_file_summaries,
        )

        if effective_use_cmg:
            rewritten_commits = []
            commit_decisions = []
            for commit in pr_context.get("commits") or []:
                rewritten = commit.get("cmg_rewritten_message")
                original = commit.get("message") or ""
                sha = commit.get("sha")
                if sha and rewritten and rewritten != original:
                    rewritten_commits.append(
                        {
                            "sha": sha,
                            "original": original,
                            "rewritten": rewritten,
                        }
                    )
                if sha:
                    commit_decisions.append(
                        {
                            "sha": sha,
                            "original": original,
                            "candidate": commit.get("cmg_candidate_message"),
                            "final": commit.get("cmg_final_message") or rewritten or original,
                            "status": commit.get("cmg_status"),
                            "reason": commit.get("cmg_reason"),
                            "patch_available": commit.get("cmg_patch_available"),
                        }
                    )
            outputs["rewritten_commits"] = rewritten_commits
            outputs["commit_decisions"] = commit_decisions

        outputs["generation_options"] = {
            "use_cmg": effective_use_cmg,
            "include_file_summaries": include_file_summaries,
            "include_commits": include_commits,
        }

        print("[ORCHESTRATOR] Successfully generated pull request description.")
        return outputs
