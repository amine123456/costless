"""Adapters for the system under test."""

from costless.config import LoadedConfig, PythonTargetSpec, SubprocessTargetSpec
from costless.targets.base import Target, TargetResponse
from costless.targets.python import PythonTarget
from costless.targets.subprocess import SubprocessTarget


def build_target(loaded: LoadedConfig) -> Target:
    spec = loaded.config.target
    match spec:
        case PythonTargetSpec():
            return PythonTarget(
                spec.callable, timeout_s=spec.timeout_s, search_path=loaded.base_dir
            )
        case SubprocessTargetSpec():
            return SubprocessTarget(
                spec.command,
                timeout_s=spec.timeout_s,
                cwd=loaded.resolve(spec.cwd) if spec.cwd else loaded.base_dir,
                env=spec.env,
            )


__all__ = ["PythonTarget", "SubprocessTarget", "Target", "TargetResponse", "build_target"]
