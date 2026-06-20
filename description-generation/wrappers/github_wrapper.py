# wrappers/github_wrapper.py
# GitHubWrapper provides methods to fetch PR-level, commit-level, and file-level information
# from a GitHub repository using the GitHub API and access token.

from github import Github, PullRequest
from typing import List, Dict, Any, Optional, Tuple
import re

ISSUE_KEYWORD_REGEX = re.compile(
    r"\b(?P<keyword>close[sd]?|fix(?:es|ed)?|resolve[sd]?)\b",
    flags=re.IGNORECASE,
)

ISSUE_REFERENCE_REGEX = re.compile(
    r"(?P<url>https?://github\.com/(?P<url_repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/issues/(?P<url_number>\d+))"
    r"|(?P<cross_repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#(?P<cross_number>\d+)"
    r"|#(?P<hash_number>\d+)",
    flags=re.IGNORECASE,
)

class GitHubWrapper:
    def __init__(self, github_token: str):
        # Initialize the GitHub API client with a personal access token
        print("[STEP] Initializing GitHub API client...\n")
        self.client = Github(github_token)

    # --- CORE PR METHODS ---

    def get_pull_request(self, repo_name: str, pr_number: int) -> PullRequest.PullRequest:
        # Fetch the pull request object from the given repo and PR number
        print(f"[STEP] Fetching pull request #{pr_number} from repository '{repo_name}'...\n")
        repo = self.client.get_repo(repo_name)
        return repo.get_pull(pr_number)

    def get_branch_name(self, pr: PullRequest.PullRequest) -> str:
        # Extract the name of the source branch from the pull request
        print("[STEP] Extracting source branch name from pull request...\n")
        return pr.head.ref

    def get_repo_metadata(self, repo_name: str) -> Dict:
        # Fetch metadata about the repo such as name, topics, license, etc.
        print(f"[STEP] Fetching metadata for repository '{repo_name}'...\n")
        repo = self.client.get_repo(repo_name)
        return {
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description,
            "language": repo.language,
            "topics": repo.get_topics(),
            "owner": repo.owner.login,
            "license": repo.license.name if repo.license else "None",
            "is_fork": repo.fork,
            "stargazers_count": repo.stargazers_count,
            "forks_count": repo.forks_count
        }

    # --- COMMIT-LEVEL METHODS ---

    def get_pull_commits(self, repo_name: str, pr_number: int) -> List[Dict]:
        # Fetch commit metadata and diff details for a pull request
        print(f"[STEP] Fetching commits for PR #{pr_number} in repo '{repo_name}'...\n")
        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commits_data = []

        for commit in pr.get_commits():
            sha = commit.sha
            message = commit.commit.message
            author = commit.commit.author.name
            author_email = commit.commit.author.email if commit.commit.author else None
            timestamp = commit.commit.author.date.isoformat()
            author_login = commit.author.login if commit.author else None
            author_avatar_url = commit.author.avatar_url if commit.author else None

            full_commit = repo.get_commit(sha)
            patches = []
            files_touched = []

            for f in full_commit.files:
                files_touched.append(f.filename)
                if hasattr(f, 'patch') and f.patch:
                    patches.append({
                        "filename": f.filename,
                        "patch": f.patch
                    })

            # Append metadata for each commit
            commits_data.append({
                "sha": sha,
                "message": message,
                "author": author,
                "author_email": author_email,
                "author_login": author_login,
                "author_avatar_url": author_avatar_url,
                "timestamp": timestamp,
                "files_touched": files_touched,
                "patches": patches,
                "is_short": len(message.strip()) < 10,
                "starts_with_verb": bool(re.match(r"^[A-Z][a-z]+", message.strip())),
            })

        print(f"[INFO] Retrieved {len(commits_data)} commits.\n")
        return commits_data

    # --- FILE-LEVEL METHODS ---

    def get_pull_files(self, repo_name: str, pr_number: int) -> List[Dict]:
        # Fetch a list of files changed in the PR along with diffs
        print(f"[STEP] Fetching file diffs for PR #{pr_number} in repo '{repo_name}'...\n")
        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        files = []

        for f in pr.get_files():
            files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch if hasattr(f, 'patch') and f.patch else "[Patch not available]"
            })

        print(f"[INFO] Retrieved {len(files)} file changes.\n")
        return files


    # --- LINKED ISSUE METHODS ---

    def get_linked_issues(self, pr: PullRequest.PullRequest) -> List[Dict]:
        """Extract linked issues from PR body and commit messages with flexible matching."""
        print("[STEP] Extracting linked issues from pull request body and commits (flexible matching)...\n")

        try:
            commit_messages = "\n".join(commit.commit.message for commit in pr.get_commits())
        except Exception:
            commit_messages = ""
        text_to_scan = commit_messages

        seen_issues: set[str] = set()
        issues: List[Dict[str, Any]] = []
        repo_cache: Dict[str, Any] = {}

        base_repo = None
        base_repo_full_name: Optional[str] = None
        try:
            base_repo = pr.base.repo if pr.base and pr.base.repo else None
            base_repo_full_name = base_repo.full_name if base_repo else None
        except Exception:
            base_repo = None
            base_repo_full_name = None

        if base_repo and base_repo_full_name:
            repo_cache[base_repo_full_name] = base_repo

        def append_issue(issue_payload: Dict[str, Any]) -> None:
            repo_full_name = issue_payload.get("repo")
            number = issue_payload.get("number")
            if not repo_full_name or number is None:
                return
            unique_key = f"{repo_full_name}#{number}"
            if unique_key in seen_issues:
                return
            seen_issues.add(unique_key)
            issues.append(issue_payload)

        for line in text_to_scan.splitlines():
            for keyword_match in ISSUE_KEYWORD_REGEX.finditer(line):
                keyword_used = keyword_match.group("keyword").lower()
                remainder = line[keyword_match.end():]
                for ref_match in ISSUE_REFERENCE_REGEX.finditer(remainder):
                    repo_name, issue_number = self._normalize_issue_reference(ref_match, base_repo_full_name)
                    if not repo_name or issue_number is None:
                        continue
                    issue_payload = self._build_issue_payload(
                        repo_name,
                        issue_number,
                        keyword_used,
                        "keyword_scan",
                        repo_cache,
                    )
                    if issue_payload:
                        append_issue(issue_payload)

        if base_repo_full_name:
            graphql_issues = self._fetch_linked_issues_graphql(base_repo_full_name, pr.number)
            for gql_issue in graphql_issues:
                append_issue(gql_issue)

        print(f"[INFO] Found {len(issues)} linked issues (flexibly parsed + GraphQL).\n")
        return issues

    # --- INTERNAL HELPERS ---

    def _fetch_linked_issues_graphql(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """Query GitHub's GraphQL API for closing and referenced issues."""
        try:
            owner, name = repo_full_name.split("/", 1)
        except ValueError:
            print(f"[WARNING] Unable to split repo name '{repo_full_name}' for GraphQL query.")
            return []

        results: List[Dict[str, Any]] = []

        for connection_name, source_label in (
            ("closingIssuesReferences", "graphql_closing"),
            ("referencedIssues", "graphql_referenced"),
        ):
            cursor: Optional[str] = None
            while True:
                query = f"""
                query($owner: String!, $name: String!, $number: Int!, $after: String) {{
                  repository(owner: $owner, name: $name) {{
                    pullRequest(number: $number) {{
                      {connection_name}(first: 50, after: $after) {{
                        nodes {{
                          number
                          title
                          body
                          url
                          state
                          repository {{ nameWithOwner }}
                        }}
                        pageInfo {{
                          hasNextPage
                          endCursor
                        }}
                      }}
                    }}
                  }}
                }}
                """.strip()

                variables = {
                    "owner": owner,
                    "name": name,
                    "number": pr_number,
                    "after": cursor,
                }

                try:
                    headers, response = self.client._Github__requester.requestJsonAndCheck(  # type: ignore[attr-defined]
                        "POST",
                        "/graphql",
                        input={"query": query, "variables": variables},
                    )
                except Exception as exc:
                    print(f"[WARNING] GraphQL query failed for {repo_full_name} PR #{pr_number}: {exc}")
                    break

                data = (
                    response.get("data", {})
                    .get("repository", {})
                    .get("pullRequest", {})
                )

                if not data:
                    break

                connection = data.get(connection_name)
                if not connection:
                    break

                for node in connection.get("nodes", []) or []:
                    if not node:
                        continue
                    issue_repo = node.get("repository", {}).get("nameWithOwner") or repo_full_name
                    issue_number = node.get("number")
                    if issue_number is None:
                        continue
                    results.append({
                        "number": issue_number,
                        "repo": issue_repo,
                        "title": node.get("title"),
                        "body": node.get("body"),
                        "state": node.get("state"),
                        "url": node.get("url"),
                        "keyword_used": None,
                        "source": source_label,
                    })

                page_info = connection.get("pageInfo") or {}
                if page_info.get("hasNextPage") and page_info.get("endCursor"):
                    cursor = page_info.get("endCursor")
                    continue
                break

        return results

    def _normalize_issue_reference(
        self,
        match: re.Match,
        default_repo: Optional[str],
    ) -> Tuple[Optional[str], Optional[int]]:
        if match.group("url"):
            return match.group("url_repo"), int(match.group("url_number"))
        if match.group("cross_repo"):
            return match.group("cross_repo"), int(match.group("cross_number"))
        if match.group("hash_number") and default_repo:
            return default_repo, int(match.group("hash_number"))
        return None, None

    def _build_issue_payload(
        self,
        repo_full_name: str,
        issue_number: int,
        keyword_used: str,
        source: str,
        repo_cache: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        repo = repo_cache.get(repo_full_name)
        if repo is None:
            try:
                repo = self.client.get_repo(repo_full_name)
                repo_cache[repo_full_name] = repo
            except Exception as exc:
                print(f"[WARNING] Could not access repository {repo_full_name}: {exc}")
                return None

        try:
            issue = repo.get_issue(issue_number)
        except Exception as exc:
            print(f"[WARNING] Could not retrieve issue {repo_full_name}#{issue_number}: {exc}")
            return None

        return {
            "number": issue.number,
            "repo": repo.full_name,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "url": issue.html_url,
            "keyword_used": keyword_used,
            "source": source,
        }
