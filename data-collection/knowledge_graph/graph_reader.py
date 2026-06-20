"""
Utilities for reading pull request context from a serialized knowledge graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from networkx.readwrite import json_graph


class KnowledgeGraphReader:
    """Loads a knowledge graph and reconstructs PR context dictionaries."""

    def __init__(self, graph_path: Path | str) -> None:
        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.graph = json_graph.node_link_graph(data, edges="links")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_pr_context(self, repo_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Return a PR context payload matching PullRequestDataCollector.collect().
        """
        pr_node_id = f"pr:{repo_name}#{pr_number}"
        if not self.graph.has_node(pr_node_id):
            # Fallback: locate by attributes when repo_name changed/redirected
            pr_node_id = None
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("label") != "PullRequest":
                    continue
                if attrs.get("number") != pr_number:
                    continue
                repo_attr = attrs.get("repo")
                if repo_attr and repo_attr.lower() == repo_name.lower():
                    pr_node_id = node_id
                    break
            if pr_node_id is None:
                raise ValueError(f"PR node not present in knowledge graph: pr:{repo_name}#{pr_number}")

        pr_node = self.graph.nodes[pr_node_id]
        pr_metadata = self._strip_label(pr_node)
        branch_name = pr_metadata.get("branch_name", "")

        repo_metadata = self._get_repo_metadata(pr_node_id)
        repo_full_name = repo_metadata.get("full_name") or repo_name

        linked_issues = self._get_linked_issues(pr_node_id)
        commits = self._get_commits(pr_node_id)
        files = self._get_files(pr_node_id, repo_full_name)

        return {
            "pr_number": pr_number,
            "repo": repo_full_name,
            "branch_name": branch_name,
            "pr_metadata": pr_metadata,
            "linked_issues": linked_issues,
            "commits": commits,
            "files": files,
            "repo_metadata": repo_metadata,
        }

    def list_pull_requests(self) -> List[tuple[str, int]]:
        """Return all pull requests stored in the knowledge graph as (repo, number)."""
        prs: List[tuple[str, int]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") != "PullRequest":
                continue
            repo = attrs.get("repo")
            number = attrs.get("number")
            if repo and isinstance(number, int):
                prs.append((repo, number))
        prs.sort()
        return prs

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_repo_metadata(self, pr_node_id: str) -> Dict[str, Any]:
        for predecessor in self.graph.predecessors(pr_node_id):
            node = self.graph.nodes[predecessor]
            if node.get("label") != "Repository":
                continue
            return self._strip_label(node)
        return {}

    def _get_linked_issues(self, pr_node_id: str) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for _, issue_id, edge_data in self.graph.edges(pr_node_id, data=True):
            if edge_data.get("label") != "PR_LINKS_ISSUE":
                continue
            issue_node = self.graph.nodes[issue_id]
            if issue_node.get("label") != "Issue":
                continue

            issue_payload = self._strip_label(issue_node)
            issue_payload["keyword_used"] = edge_data.get("keyword")
            issue_payload["source"] = edge_data.get("origin")
            issues.append(issue_payload)

        return issues

    def _get_commits(self, pr_node_id: str) -> List[Dict[str, Any]]:
        commits: List[Dict[str, Any]] = []
        for _, commit_id, edge_data in self.graph.edges(pr_node_id, data=True):
            if edge_data.get("label") != "PR_CONTAINS_COMMIT":
                continue
            commit_node = self.graph.nodes[commit_id]
            if commit_node.get("label") != "Commit":
                continue

            commit_payload = self._strip_label(commit_node)
            # Ensure expected fields are present, even if empty.
            commit_payload.setdefault("patches", [])
            commit_payload.setdefault("files_touched", [])
            commits.append(commit_payload)

        return commits

    def _get_files(self, pr_node_id: str, repo_full_name: str) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        for _, file_id, edge_data in self.graph.edges(pr_node_id, data=True):
            if edge_data.get("label") != "PR_MODIFIES_FILE":
                continue
            file_node = self.graph.nodes[file_id]
            if file_node.get("label") != "File":
                continue

            filename = file_node.get("path") or file_node.get("name")
            patch_content = edge_data.get("patch")
            if patch_content is None:
                patch_content = "[Patch not available]"

            file_payload = {
                "filename": filename,
                "status": edge_data.get("status"),
                "additions": edge_data.get("additions"),
                "deletions": edge_data.get("deletions"),
                "changes": edge_data.get("changes"),
                "patch": patch_content,
                "repo": repo_full_name,
            }
            files.append(file_payload)

        return files

    @staticmethod
    def _strip_label(attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of node attributes without the `label` key."""
        return {k: v for k, v in attrs.items() if k != "label"}
