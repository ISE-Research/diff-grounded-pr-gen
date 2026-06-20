"""
Knowledge graph construction utilities for pull-request data.

The graph is represented as a NetworkX MultiDiGraph where nodes carry a `label`
attribute indicating their entity type (e.g., Repository, PullRequest, Commit),
and edges capture semantic relationships between the entities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import networkx as nx
from networkx.readwrite import json_graph

from components.cmg_quality import compute_commit_quality_annotations

class KnowledgeGraphBuilder:
    """Builds and persists a knowledge graph from collected PR data."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_pr_data(self, pr_data: Dict[str, Any]) -> None:
        """Add nodes and edges for a single pull request."""
        repo_name = pr_data.get("repo") or pr_data.get("repo_name")
        repo_metadata = pr_data.get("repo_metadata", {}) or {}
        pr_metadata = pr_data.get("pr_metadata", {}) or {}
        branch_name = pr_data.get("branch_name")
        pr_number = pr_metadata.get("number") or pr_data.get("pr_number")

        if not repo_name or pr_number is None:
            raise ValueError("Missing repo name or PR number in collected data.")

        repo_id = f"repo:{repo_metadata.get('full_name') or repo_name}"
        pr_id = f"pr:{repo_name}#{pr_number}"

        # Repository node
        repo_attrs = _filter_none({
            "label": "Repository",
            "name": repo_metadata.get("name"),
            "full_name": repo_metadata.get("full_name") or repo_name,
            "description": repo_metadata.get("description"),
            "language": repo_metadata.get("language"),
        })
        self._upsert_node(repo_id, repo_attrs)

        # Pull Request node
        pr_attrs = _filter_none({
            "label": "PullRequest",
            "repo": repo_metadata.get("full_name") or repo_name,
            "number": pr_number,
            "title": pr_metadata.get("title"),
            "state": pr_metadata.get("state"),
            "is_draft": pr_metadata.get("is_draft"),
            "merge_commit_sha": pr_metadata.get("merge_commit_sha"),
            "branch_name": branch_name,
            "base_branch": pr_metadata.get("base_branch"),
            "base_repo": pr_metadata.get("base_repo"),
            "head_branch": pr_metadata.get("head_branch"),
            "additions": pr_metadata.get("additions"),
            "deletions": pr_metadata.get("deletions"),
            "changed_files": pr_metadata.get("changed_files"),
            "labels": pr_metadata.get("labels"),
        })
        self._upsert_node(pr_id, pr_attrs)

        self._add_edge(repo_id, pr_id, "REPO_HAS_PR")

        # Linked issues
        for issue in pr_data.get("linked_issues", []) or []:
            issue_repo = issue.get("repo") or repo_name
            issue_number = issue.get("number")
            if issue_number is None:
                continue

            issue_id = f"issue:{issue_repo}#{issue_number}"
            issue_attrs = _filter_none({
                "label": "Issue",
                "repo": issue_repo,
                "number": issue_number,
                "title": issue.get("title"),
                "body": issue.get("body"),
                "state": issue.get("state"),
                "url": issue.get("url"),
                "source": issue.get("source"),
            })
            self._upsert_node(issue_id, issue_attrs)
            self._add_edge(
                pr_id,
                issue_id,
                "PR_LINKS_ISSUE",
                keyword=issue.get("keyword_used"),
                origin=issue.get("source"),
            )

        # Commits
        for commit in pr_data.get("commits", []) or []:
            commit_sha = commit.get("sha")
            if not commit_sha:
                continue

            quality_ann = compute_commit_quality_annotations(
                msg=commit.get("message"),
                patches=commit.get("patches"),
                files_touched=commit.get("files_touched"),
            )

            commit_id = f"commit:{commit_sha}"
            commit_attrs = _filter_none({
                "label": "Commit",
                "sha": commit_sha,
                "message": commit.get("message"),
                "timestamp": commit.get("timestamp"),
                "author_name": commit.get("author"),
                "author_login": commit.get("author_login"),
                "files_touched": commit.get("files_touched"),
                "patches": commit.get("patches"),
                **quality_ann,
            })
            self._upsert_node(commit_id, commit_attrs)
            self._add_edge(pr_id, commit_id, "PR_CONTAINS_COMMIT")

        # File-level summaries for PR
        for file_info in pr_data.get("files", []) or []:
            filename = file_info.get("filename")
            if not filename:
                continue

            file_id = f"file:{repo_name}:{filename}"
            file_attrs = _filter_none({
                "label": "File",
                "repo": repo_name,
                "path": filename,
            })
            self._upsert_node(file_id, file_attrs)

            edge_attrs = _filter_none({
                "status": file_info.get("status"),
                "additions": file_info.get("additions"),
                "deletions": file_info.get("deletions"),
                "changes": file_info.get("changes"),
                "patch": file_info.get("patch"),
            })
            self._add_edge(pr_id, file_id, "PR_MODIFIES_FILE", **edge_attrs)

        # Store PR body as a text attribute on the node (optional)
        if pr_metadata.get("body"):
            existing_body = self.graph.nodes[pr_id].get("body")
            if not existing_body:
                self.graph.nodes[pr_id]["body"] = pr_metadata["body"]

    def ingest_many(self, pr_iterable: Iterable[Dict[str, Any]]) -> None:
        """Bulk ingest multiple PR payloads."""
        for pr_data in pr_iterable:
            self.ingest_pr_data(pr_data)

    def to_node_link_data(self) -> Dict[str, Any]:
        """Return the graph in NetworkX node-link JSON format."""
        return json_graph.node_link_data(self.graph, edges="links")

    def save_json(self, path: Path | str, indent: Optional[int] = 2) -> Path:
        """Serialize the graph to JSON (node-link format)."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_node_link_data(), f, indent=indent)
        return output_path

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _upsert_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, **attrs)

    def _add_edge(self, source_id: str, target_id: str, label: str, **attrs: Any) -> None:
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return

        attrs = _filter_none({"label": label, **attrs})
        existing_edges = self.graph.get_edge_data(source_id, target_id, default={})

        # Prevent duplicate edges with identical labels and attributes.
        for edge_data in existing_edges.values():
            if edge_data.get("label") == label:
                # If all provided attrs already match, skip; otherwise allow parallel edge.
                if all(edge_data.get(k) == attrs.get(k) for k in attrs):
                    return
        self.graph.add_edge(source_id, target_id, **attrs)


def _filter_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values to keep graph metadata tidy."""
    return {k: v for k, v in payload.items() if v is not None}

