import io
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from typer.testing import CliRunner

from costless.ci import CIError
from costless.ci.platforms import GITHUB_COMMENT_LIMIT, GitHub, GitLab, detect_platform
from costless.cli import app
from costless.report import MARKER
from costless.results import write_run
from tests.builders import make_run
from tests.conftest import WriteFile

REPORT = f"{MARKER}\n## report\n"
NEXT_PAGE = "https://api.github.com/repos/acme/app/issues/12/comments?per_page=100&page=2"


@pytest.fixture(autouse=True)
def _not_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests run inside CI too; never let them talk to the real platform.
    for var in ("GITLAB_CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)


GITLAB_ENV = {
    "GITLAB_CI": "true",
    "CI_API_V4_URL": "https://gitlab.example.com/api/v4",
    "CI_PROJECT_ID": "42",
    "CI_MERGE_REQUEST_IID": "7",
    "CI_MERGE_REQUEST_TARGET_BRANCH_NAME": "main",
    "CI_JOB_TOKEN": "job-token",
    "COSTLESS_GITLAB_TOKEN": "api-token",
}


Handler = Callable[[httpx2.Request], httpx2.Response]


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx2.Request] = []
        self._handler = handler

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._handler(request)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)


class TestGitLab:
    def test_detected_from_env(self) -> None:
        platform = detect_platform(GITLAB_ENV)
        assert isinstance(platform, GitLab)
        assert platform.baseline_ref == "main"
        assert platform.baseline_job == "costless-baseline"
        assert detect_platform({}) is None

    def test_fetch_baseline(self, tmp_path: Path) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert (
                request.url.path == "/api/v4/projects/42/jobs/artifacts/main/raw/.costless/run.json"
            )
            assert request.url.params["job"] == "costless-baseline"
            assert request.headers["PRIVATE-TOKEN"] == "api-token"
            return httpx2.Response(200, content=b'{"schema_version": 1}')

        platform = GitLab.from_env(GITLAB_ENV, transport=Recorder(handler).transport())
        assert platform.fetch_baseline(tmp_path / "b.json")
        assert (tmp_path / "b.json").read_text() == '{"schema_version": 1}'

    def test_fetch_baseline_with_job_token_and_none_available(self, tmp_path: Path) -> None:
        env = {k: v for k, v in GITLAB_ENV.items() if k != "COSTLESS_GITLAB_TOKEN"}

        def handler(request: httpx2.Request) -> httpx2.Response:
            assert request.headers["JOB-TOKEN"] == "job-token"
            return httpx2.Response(404, json={"message": "404 Not Found"})

        platform = GitLab.from_env(env, transport=Recorder(handler).transport())
        assert not platform.fetch_baseline(tmp_path / "b.json")

    def test_creates_comment_when_none_exists(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                return httpx2.Response(200, json=[{"id": 1, "body": "LGTM"}])
            assert request.method == "POST"
            assert request.url.path == "/api/v4/projects/42/merge_requests/7/notes"
            assert json.loads(request.content) == {"body": REPORT}
            return httpx2.Response(201, json={"id": 99})

        rec = Recorder(handler)
        assert GitLab.from_env(GITLAB_ENV, transport=rec.transport()).upsert_comment(REPORT) == "99"

    def test_updates_existing_comment_found_on_a_later_page(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                page = request.url.params["page"]
                if page == "1":
                    return httpx2.Response(
                        200, json=[{"id": 1, "body": "hi"}], headers={"x-next-page": "2"}
                    )
                return httpx2.Response(200, json=[{"id": 5, "body": f"{MARKER}\nold"}])
            assert request.method == "PUT"
            assert request.url.path.endswith("/notes/5")
            return httpx2.Response(200, json={"id": 5})

        rec = Recorder(handler)
        GitLab.from_env(GITLAB_ENV, transport=rec.transport()).upsert_comment(REPORT)
        assert [r.method for r in rec.requests] == ["GET", "GET", "PUT"]

    def test_comment_needs_an_api_token(self) -> None:
        env = {k: v for k, v in GITLAB_ENV.items() if k != "COSTLESS_GITLAB_TOKEN"}
        with pytest.raises(CIError, match="needs COSTLESS_GITLAB_TOKEN"):
            GitLab.from_env(env).upsert_comment(REPORT)

    def test_api_errors_are_reported(self) -> None:
        transport = httpx2.MockTransport(lambda r: httpx2.Response(403, text="forbidden"))
        with pytest.raises(CIError, match="listing MR comments failed: HTTP 403 forbidden"):
            GitLab.from_env(GITLAB_ENV, transport=transport).upsert_comment(REPORT)

    def test_missing_variables(self) -> None:
        with pytest.raises(CIError, match="CI_API_V4_URL is not set"):
            GitLab.from_env({"GITLAB_CI": "true"})


def github_env(tmp_path: Path) -> dict[str, str]:
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 12}}))
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "acme/app",
        "GITHUB_TOKEN": "gh-token",
        "GITHUB_BASE_REF": "main",
        "GITHUB_EVENT_PATH": str(event),
    }


def zipped(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(name, content)
    return buffer.getvalue()


class TestGitHub:
    def test_detected_from_env(self, tmp_path: Path) -> None:
        platform = detect_platform(github_env(tmp_path))
        assert isinstance(platform, GitHub)
        assert platform.pull_request == 12

    def test_fetch_baseline_picks_newest_artifact_on_base_branch(self, tmp_path: Path) -> None:
        artifacts = {
            "artifacts": [
                {
                    "id": 1,
                    "expired": False,
                    "created_at": "2026-01-03",
                    "archive_download_url": "https://api.github.com/a/1",
                    "workflow_run": {"head_branch": "feature"},
                },
                {
                    "id": 2,
                    "expired": False,
                    "created_at": "2026-01-02",
                    "archive_download_url": "https://api.github.com/a/2",
                    "workflow_run": {"head_branch": "main"},
                },
                {
                    "id": 3,
                    "expired": True,
                    "created_at": "2026-01-04",
                    "archive_download_url": "https://api.github.com/a/3",
                    "workflow_run": {"head_branch": "main"},
                },
                {
                    "id": 4,
                    "expired": False,
                    "created_at": "2026-01-01",
                    "archive_download_url": "https://api.github.com/a/4",
                    "workflow_run": {"head_branch": "main"},
                },
            ]
        }

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/repos/acme/app/actions/artifacts":
                assert request.url.params["name"] == "costless-baseline"
                assert request.headers["Authorization"] == "Bearer gh-token"
                return httpx2.Response(200, json=artifacts)
            if request.url.path == "/a/2":
                return httpx2.Response(302, headers={"location": "https://blob.example.com/zip"})
            assert request.url.host == "blob.example.com"
            assert "Authorization" not in request.headers  # never leak the token to blob storage
            return httpx2.Response(200, content=zipped("run.json", b'{"ok": true}'))

        platform = GitHub.from_env(github_env(tmp_path), transport=Recorder(handler).transport())
        assert platform.fetch_baseline(tmp_path / "b.json")
        assert (tmp_path / "b.json").read_text() == '{"ok": true}'

    def test_no_baseline_artifact_yet(self, tmp_path: Path) -> None:
        transport = httpx2.MockTransport(lambda r: httpx2.Response(200, json={"artifacts": []}))
        assert not GitHub.from_env(github_env(tmp_path), transport=transport).fetch_baseline(
            tmp_path / "b"
        )

    def test_upsert_comment_follows_pagination_and_updates(self, tmp_path: Path) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET" and "page=2" not in str(request.url):
                return httpx2.Response(
                    200,
                    json=[{"id": 1, "body": "nice"}],
                    headers={"link": f'<{NEXT_PAGE}>; rel="next"'},
                )
            if request.method == "GET":
                return httpx2.Response(200, json=[{"id": 8, "body": f"{MARKER}\nold"}])
            assert request.method == "PATCH"
            assert request.url.path == "/repos/acme/app/issues/comments/8"
            return httpx2.Response(
                200, json={"id": 8, "html_url": "https://github.com/acme/app/pull/12#c8"}
            )

        platform = GitHub.from_env(github_env(tmp_path), transport=Recorder(handler).transport())
        assert platform.upsert_comment(REPORT) == "https://github.com/acme/app/pull/12#c8"

    def test_creates_comment_and_truncates_huge_reports(self, tmp_path: Path) -> None:
        posted: list[str] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                return httpx2.Response(200, json=[])
            posted.append(json.loads(request.content)["body"])
            return httpx2.Response(201, json={"id": 3})

        platform = GitHub.from_env(github_env(tmp_path), transport=Recorder(handler).transport())
        platform.upsert_comment(REPORT + "x" * 100_000)
        assert len(posted[0]) == GITHUB_COMMENT_LIMIT
        assert posted[0].startswith(MARKER)
        assert posted[0].endswith("the full report is in the job artifacts._\n")

    def test_requires_token_and_pull_request(self, tmp_path: Path) -> None:
        env = github_env(tmp_path)
        with pytest.raises(CIError, match="GITHUB_TOKEN is not set"):
            GitHub.from_env({k: v for k, v in env.items() if k != "GITHUB_TOKEN"})
        push = GitHub.from_env({k: v for k, v in env.items() if k != "GITHUB_EVENT_PATH"})
        with pytest.raises(CIError, match="not a pull request"):
            push.upsert_comment(REPORT)


# ------------------------------------------------------------------- ci check


def stable(value: float) -> dict[str, list[float]]:
    return {f"c{i:02d}": [value] * 5 for i in range(20)}


def config(write: WriteFile, extra: str = "") -> str:
    write("cases.yaml", "- {id: a, input: x, expected: x}\n")
    return str(
        write(
            "costless.yaml",
            f"""
            version: 1
            target: {{type: python, callable: "unused:run"}}
            datasets: [{{path: cases.yaml}}]
            scorers: [{{type: exact_match}}]
            {extra}
            """,
        )
    )


def test_ci_check_passes_and_writes_artifacts(write: WriteFile, tmp_path: Path) -> None:
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    write_run(make_run(stable(1.0), ref="main"), base)
    write_run(make_run(stable(1.0)), cand)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "ci",
            "check",
            "-c",
            config(write),
            "--candidate",
            str(cand),
            "-b",
            str(base),
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "report.md").read_text().startswith(MARKER)
    assert json.loads((out / "comparison.json").read_text())["gate_passed"]
    assert (out / "run.json").exists()


def test_ci_check_fails_on_regression(write: WriteFile, tmp_path: Path) -> None:
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    write_run(make_run(stable(1.0), ref="main"), base)
    write_run(make_run(stable(0.0)), cand)
    result = CliRunner().invoke(
        app,
        [
            "ci",
            "check",
            "-c",
            config(write),
            "--candidate",
            str(cand),
            "-b",
            str(base),
            "--out-dir",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 1
    assert "quality gate failed" in result.stdout


def test_ci_check_budget_failure_is_part_of_the_report(write: WriteFile, tmp_path: Path) -> None:
    cand = tmp_path / "cand.json"
    write_run(make_run(stable(1.0), cost_usd=0.5), cand)
    result = CliRunner().invoke(
        app,
        [
            "ci",
            "check",
            "-c",
            config(write, "budget: {max_case_usd: 0.01}"),
            "--candidate",
            str(cand),
            "--out-dir",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 1
    report = (tmp_path / "o" / "report.md").read_text()
    assert "❌ costless: quality gate failed" in report
    assert re.search(
        r"- budget: \[max_case_usd\] mean cost per case \$0\.500000 exceeds \$0\.01", report
    )
    assert "No baseline run available" in report


def test_ci_commands_outside_ci(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ci", "fetch-baseline", "-o", str(tmp_path / "b.json")])
    assert result.exit_code == 2
    assert "not running in GitLab CI or GitHub Actions" in result.stderr
