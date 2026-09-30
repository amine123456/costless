import hashlib

import pytest

from costless.dataset import load_dataset
from costless.errors import DatasetError
from tests.conftest import WriteFile


def test_yaml_mapping(write: WriteFile) -> None:
    path = write(
        "cases.yaml",
        """
        dataset: incidents
        version: 3
        cases:
          - id: inc-001
            input: {text: "db down"}
            expected: {severity: SEV1}
            tags: [database]
          - id: inc-002
            input: "api slow"
            rubric: "mentions latency"
        """,
    )
    ds = load_dataset(path)

    assert ds.name == "incidents"
    assert ds.version == "3"
    assert [c.id for c in ds.cases] == ["inc-001", "inc-002"]
    assert ds.cases[0].tags == ("database",)
    assert ds.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert ds.info().cases == 2


def test_yaml_list_uses_file_stem(write: WriteFile) -> None:
    ds = load_dataset(write("smoke.yml", "- {id: a, input: x}\n"))
    assert (ds.name, ds.version) == ("smoke", None)


def test_jsonl(write: WriteFile) -> None:
    ds = load_dataset(write("golden.jsonl", '{"id": "a", "input": 1}\n\n{"id": "b", "input": 2}\n'))
    assert ds.name == "golden"
    assert [c.input for c in ds.cases] == [1, 2]


def test_case_level_scorers_are_validated(write: WriteFile) -> None:
    ds = load_dataset(
        write("s.yaml", "- {id: a, input: x, scorers: [{type: regex, pattern: 'x'}]}\n")
    )
    assert ds.cases[0].scorers[0].type == "regex"


def test_filter_tags(write: WriteFile) -> None:
    ds = load_dataset(
        write(
            "t.yaml",
            """
            - {id: a, input: x, tags: [db]}
            - {id: b, input: x, tags: [net]}
            - {id: c, input: x}
            """,
        )
    )
    assert [c.id for c in ds.filter_tags(["db", "net"]).cases] == ["a", "b"]
    assert ds.filter_tags([]) is ds


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("dup.yaml", "- {id: a, input: x}\n- {id: a, input: y}\n", "duplicate case id 'a'"),
        ("bad-id.yaml", "- {id: 'has space', input: x}\n", "case has space"),
        ("extra.yaml", "- {id: a, input: x, expectd: 1}\n", "Extra inputs are not permitted"),
        ("keys.yaml", "cases: []\nowner: me\n", "unknown top-level keys: owner"),
        ("empty.yaml", "cases: []\n", "dataset has no cases"),
        ("scalar.yaml", "hello\n", "expected a list of cases"),
        ("broken.yaml", "cases: [\n", "invalid YAML"),
        ("broken.jsonl", '{"id": "a", "input": 1}\n{oops\n', "broken.jsonl:2: invalid JSON"),
        ("cases.csv", "id,input\n", "unsupported dataset format '.csv'"),
    ],
)
def test_invalid_datasets(write: WriteFile, name: str, content: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(write(name, content))


def test_missing_file(write: WriteFile) -> None:
    path = write("x.yaml", "- {id: a, input: x}\n")
    with pytest.raises(DatasetError, match="dataset not found"):
        load_dataset(path.with_name("missing.yaml"))
