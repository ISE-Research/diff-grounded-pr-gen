# components/data_collector.py
# PullRequestDataCollector gathers all essential inputs for PR description generation,
# including commits, files, issues, repo metadata, and branch info.

class PullRequestDataCollector:
    def __init__(self, github_wrapper):
        self.wrapper = github_wrapper

    def collect(self, repo: str, pr_number: int) -> dict:
        print(f"[PullRequestDataCollector] Collecting data for PR #{pr_number} in {repo}...\n")

        # === PR Metadata ===
        print("[PullRequestDataCollector] Fetching PR metadata and branch name...\n")
        pr = self.wrapper.get_pull_request(repo, pr_number)
        branch_name = self.wrapper.get_branch_name(pr)
        labels = [lbl.name for lbl in getattr(pr, "labels", [])] if hasattr(pr, "labels") else []
        pr_metadata = {
            "id": pr.id,
            "number": pr.number,
            "title": pr.title or "",
            "body": pr.body or "",
            "state": pr.state,
            "is_draft": pr.draft,
            "created_at": pr.created_at.isoformat() if pr.created_at else None,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
            "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
            "merge_commit_sha": pr.merge_commit_sha,
            "html_url": pr.html_url,
            "author_login": pr.user.login if pr.user else None,
            "author_name": pr.user.name if pr.user and hasattr(pr.user, "name") else None,
            "author_avatar_url": pr.user.avatar_url if pr.user else None,
            "base_branch": pr.base.ref if pr.base else None,
            "base_repo": pr.base.repo.full_name if pr.base and pr.base.repo else None,
            "head_branch": pr.head.ref if pr.head else None,
            "head_repo": pr.head.repo.full_name if pr.head and pr.head.repo else None,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "labels": labels,
        }

        # === Commits + Diffs ===
        print("[PullRequestDataCollector] Fetching commit messages and patches...\n")
        commits = self.wrapper.get_pull_commits(repo, pr_number)

        # === File-level Diffs ===
        print("[PullRequestDataCollector] Fetching file-level code diffs...\n")
        files = self.wrapper.get_pull_files(repo, pr_number)

        # === Linked Issues ===
        print("[PullRequestDataCollector] Extracting linked issues from PR body...\n")
        linked_issues = self.wrapper.get_linked_issues(pr)

        # === Repo Metadata ===
        print("[PullRequestDataCollector] Fetching repository metadata...\n")
        repo_metadata = self.wrapper.get_repo_metadata(repo)

        print(f"[PullRequestDataCollector] Collected: {len(commits)} commits, {len(files)} files, {len(linked_issues)} issues.\n")

        return {
            "pr_number": pr_number,
            "repo": repo,
            "branch_name": branch_name,
            "pr_metadata": pr_metadata,
            "linked_issues": linked_issues,
            "commits": commits,
            "files": files,
            "repo_metadata": repo_metadata,
        }
