"""GitLab and GitHub clients, configured from the CI job's own environment.

Both platforms do two things for costless:

- **fetch the baseline**: the ``run.json`` produced by the latest successful
  eval job on the target branch (a job artifact on GitLab, a workflow artifact
  on GitHub);
- **upsert the report**: create the MR/PR comment on the first run and edit the
  same comment afterwards, found by a hidden marker, so pushes never pile up
  comments.
"""

import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx2

from costless.errors import CostlessError
from costless.report import MARKER

GITHUB_COMMENT_LIMIT = 65_536
GITLAB_NOTE_LIMIT = 1_000_000
_TRUNCATION_NOTE = "\n\n_Report truncated; the full report is in the job artifacts._\n"


class CIError(CostlessError):
    """A CI platform API call failed."""


class CIPlatform(Protocol):
    name: str

    def fetch_baseline(self, dest: Path) -> bool:
        """Write the baseline run.json to ``dest``. False if there is none yet."""
        ...

    def upsert_comment(self, body: str) -> str:
        """Create or update the report comment; return its URL or id."""
        ...


def detect_platform(
    env: Mapping[str, str] | None = None, *, transport: httpx2.BaseTransport | None = None
) -> CIPlatform | None:
    env = os.environ if env is None else env
    if env.get("GITLAB_CI") == "true":
        return GitLab.from_env(env, transport=transport)
    if env.get("GITHUB_ACTIONS") == "true":
        return GitHub.from_env(env, transport=transport)
    return None


def _truncate(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body
    return body[: limit - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


def _check(response: httpx2.Response, action: str) -> None:
    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        msg = f"{action} failed: HTTP {response.status_code} {detail}"
        raise CIError(msg)


# --------------------------------------------------------------------------- GitLab


@dataclass
class GitLab:
    api_url: str
    project_id: str
    baseline_ref: str
    baseline_job: str
    baseline_path: str
    job_token: str | None
    api_token: str | None
    merge_request_iid: str | None
    transport: httpx2.BaseTransport | None = None
    name: str = "gitlab"

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, transport: httpx2.BaseTransport | None = None
    ) -> "GitLab":
        try:
            api_url = env["CI_API_V4_URL"]
            project_id = env["CI_PROJECT_ID"]
        except KeyError as exc:
            msg = f"GitLab CI variable {exc.args[0]} is not set"
            raise CIError(msg) from exc
        return cls(
            api_url=api_url.rstrip("/"),
            project_id=project_id,
            baseline_ref=env.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME")
            or env.get("CI_DEFAULT_BRANCH")
            or "main",
            baseline_job=env.get("COSTLESS_BASELINE_JOB", "costless-baseline"),
            baseline_path=env.get("COSTLESS_BASELINE_PATH", ".costless/run.json"),
            job_token=env.get("CI_JOB_TOKEN"),
            api_token=env.get("COSTLESS_GITLAB_TOKEN"),
            merge_request_iid=env.get("CI_MERGE_REQUEST_IID"),
            transport=transport,
        )

    def _client(self, *, for_write: bool) -> httpx2.Client:
        if self.api_token:
            headers = {"PRIVATE-TOKEN": self.api_token}
        elif self.job_token and not for_write:
            headers = {"JOB-TOKEN": self.job_token}
        elif for_write:
            # CI_JOB_TOKEN cannot write MR notes; a project access token is required.
            msg = "posting the MR comment needs COSTLESS_GITLAB_TOKEN (a token with api scope)"
            raise CIError(msg)
        else:
            msg = "no GitLab credentials: CI_JOB_TOKEN or COSTLESS_GITLAB_TOKEN"
            raise CIError(msg)
        return httpx2.Client(
            base_url=self.api_url, headers=headers, timeout=60, transport=self.transport
        )

    def fetch_baseline(self, dest: Path) -> bool:
        url = (
            f"/projects/{quote(self.project_id, safe='')}/jobs/artifacts/"
            f"{quote(self.baseline_ref, safe='')}/raw/{self.baseline_path}"
        )
        with self._client(for_write=False) as client:
            response = client.get(url, params={"job": self.baseline_job}, follow_redirects=True)
        if response.status_code == 404:
            return False
        _check(response, f"downloading the baseline from {self.baseline_ref}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return True

    def upsert_comment(self, body: str) -> str:
        if not self.merge_request_iid:
            msg = "not a merge request pipeline (CI_MERGE_REQUEST_IID is not set)"
            raise CIError(msg)
        body = _truncate(body, GITLAB_NOTE_LIMIT)
        notes = (
            f"/projects/{quote(self.project_id, safe='')}"
            f"/merge_requests/{self.merge_request_iid}/notes"
        )
        with self._client(for_write=True) as client:
            existing = next(
                (n for n in self._notes(client, notes) if MARKER in (n.get("body") or "")), None
            )
            if existing is None:
                response = client.post(notes, json={"body": body})
                _check(response, "creating the MR comment")
            else:
                response = client.put(f"{notes}/{existing['id']}", json={"body": body})
                _check(response, "updating the MR comment")
        return str(response.json().get("id"))

    def _notes(self, client: httpx2.Client, url: str) -> Iterator[dict[str, Any]]:
        page: str | None = "1"
        while page:
            response = client.get(
                url, params={"per_page": 100, "page": page, "sort": "asc", "order_by": "created_at"}
            )
            _check(response, "listing MR comments")
            yield from response.json()
            page = response.headers.get("x-next-page") or None


# --------------------------------------------------------------------------- GitHub


@dataclass
class GitHub:
    api_url: str
    repository: str
    token: str
    baseline_ref: str
    artifact_name: str
    pull_request: int | None
    transport: httpx2.BaseTransport | None = None
    name: str = "github"

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, transport: httpx2.BaseTransport | None = None
    ) -> "GitHub":
        token = env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")
        if not token:
            msg = "GITHUB_TOKEN is not set (pass `env: GITHUB_TOKEN: ${{ github.token }}`)"
            raise CIError(msg)
        repository = env.get("GITHUB_REPOSITORY")
        if not repository:
            msg = "GITHUB_REPOSITORY is not set"
            raise CIError(msg)
        return cls(
            api_url=env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
            repository=repository,
            token=token,
            baseline_ref=env.get("GITHUB_BASE_REF") or env.get("COSTLESS_BASELINE_REF") or "main",
            artifact_name=env.get("COSTLESS_BASELINE_ARTIFACT", "costless-baseline"),
            pull_request=_pull_request_number(env),
            transport=transport,
        )

    def _client(self) -> httpx2.Client:
        return httpx2.Client(
            base_url=self.api_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=60,
            transport=self.transport,
        )

    def fetch_baseline(self, dest: Path) -> bool:
        with self._client() as client:
            response = client.get(
                f"/repos/{self.repository}/actions/artifacts",
                params={"name": self.artifact_name, "per_page": 100},
            )
            _check(response, "listing workflow artifacts")
            candidates = [
                a
                for a in response.json().get("artifacts", [])
                if not a.get("expired")
                and (a.get("workflow_run") or {}).get("head_branch") == self.baseline_ref
            ]
            if not candidates:
                return False
            newest = max(candidates, key=lambda a: a.get("created_at", ""))
            # The download URL redirects to blob storage; httpx drops the
            # Authorization header when the redirect leaves the API host.
            archive = client.get(newest["archive_download_url"], follow_redirects=True)
            _check(archive, "downloading the baseline artifact")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            members = [m for m in bundle.namelist() if m.endswith("run.json")]
            if not members:
                msg = f"artifact {self.artifact_name!r} contains no run.json"
                raise CIError(msg)
            data = bundle.read(sorted(members, key=len)[0])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True

    def upsert_comment(self, body: str) -> str:
        if self.pull_request is None:
            msg = "not a pull request event (no pull request number found)"
            raise CIError(msg)
        body = _truncate(body, GITHUB_COMMENT_LIMIT)
        with self._client() as client:
            existing = next(
                (c for c in self._comments(client) if MARKER in (c.get("body") or "")), None
            )
            if existing is None:
                response = client.post(
                    f"/repos/{self.repository}/issues/{self.pull_request}/comments",
                    json={"body": body},
                )
                _check(response, "creating the PR comment")
            else:
                response = client.patch(
                    f"/repos/{self.repository}/issues/comments/{existing['id']}",
                    json={"body": body},
                )
                _check(response, "updating the PR comment")
        return str(response.json().get("html_url") or response.json().get("id"))

    def _comments(self, client: httpx2.Client) -> Iterator[dict[str, Any]]:
        url: str | None = f"/repos/{self.repository}/issues/{self.pull_request}/comments"
        params: dict[str, Any] | None = {"per_page": 100}
        while url:
            response = client.get(url, params=params)
            _check(response, "listing PR comments")
            yield from response.json()
            url = response.links.get("next", {}).get("url")
            params = None  # the next link already carries the query


def _pull_request_number(env: Mapping[str, str]) -> int | None:
    explicit = env.get("COSTLESS_PR_NUMBER")
    if explicit:
        return int(explicit)
    event_path = env.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    number = (event.get("pull_request") or {}).get("number")
    return int(number) if number is not None else None
