"""Build a provider from environment variables, so the same app code runs
against Anthropic, xAI, any OpenAI-compatible endpoint, or a replay file.

=====================  ==================================================
Variable               Meaning
=====================  ==================================================
COSTLESS_PROVIDER      anthropic (default) | gemini | xai | openai | replay
COSTLESS_MODEL         model id; defaults per provider (see below)
COSTLESS_BASE_URL      override the endpoint of xai / openai
COSTLESS_REPLAY_FILE   recording to replay (provider ``replay``)
COSTLESS_RECORD_FILE   record every real call to this JSONL file
COSTLESS_MAX_RETRIES   default 3
COSTLESS_TIMEOUT_S     per-request timeout, default 120
COSTLESS_JUDGE_PROVIDER  provider for LLM judges (default: COSTLESS_PROVIDER)
COSTLESS_JUDGE_MODEL     model for LLM judges
=====================  ==================================================

Credentials: ``ANTHROPIC_API_KEY`` (or any credential the Anthropic SDK
resolves), ``GEMINI_API_KEY``, ``XAI_API_KEY``, ``OPENAI_API_KEY``.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from costless.errors import ConfigError
from costless.providers.anthropic import AnthropicProvider
from costless.providers.base import Provider
from costless.providers.openai_compat import OpenAICompatibleProvider
from costless.providers.replay import RecordingProvider, ReplayProvider

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-3.5-flash",
    "xai": "grok-4.7",
}

_OPENAI_COMPATIBLE = {
    # name: (default base URL, API key variable)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def provider_from_env(env: Mapping[str, str] | None = None) -> Provider:
    env = os.environ if env is None else env
    name = env.get("COSTLESS_PROVIDER", "anthropic").strip().lower()
    max_retries = _int(env, "COSTLESS_MAX_RETRIES", 3)
    timeout_s = _float(env, "COSTLESS_TIMEOUT_S", 120.0)

    provider: Provider
    if name == "anthropic":
        provider = AnthropicProvider(max_retries=max_retries, timeout_s=timeout_s)
    elif name in _OPENAI_COMPATIBLE:
        default_url, key_var = _OPENAI_COMPATIBLE[name]
        api_key = env.get(key_var, "").strip()
        if not api_key:
            msg = f"COSTLESS_PROVIDER={name} needs {key_var} to be set"
            raise ConfigError(msg)
        provider = OpenAICompatibleProvider(
            name=name,
            base_url=env.get("COSTLESS_BASE_URL") or default_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
    elif name == "replay":
        replay_file = env.get("COSTLESS_REPLAY_FILE")
        if not replay_file:
            msg = "COSTLESS_PROVIDER=replay needs COSTLESS_REPLAY_FILE"
            raise ConfigError(msg)
        return ReplayProvider(Path(replay_file))
    else:
        known = ", ".join(["anthropic", *_OPENAI_COMPATIBLE, "replay"])
        msg = f"unknown COSTLESS_PROVIDER {name!r} (expected one of: {known})"
        raise ConfigError(msg)

    record_file = env.get("COSTLESS_RECORD_FILE")
    if record_file:
        return RecordingProvider(provider, Path(record_file))
    return provider


def judge_provider_and_model(
    provider: str | None, model: str | None, env: Mapping[str, str] | None = None
) -> tuple[Provider, str]:
    """Provider and model for an LLM judge.

    Precedence: the judge's own config, then COSTLESS_JUDGE_PROVIDER /
    COSTLESS_JUDGE_MODEL, then the target's COSTLESS_PROVIDER / COSTLESS_MODEL.
    """
    env = dict(os.environ if env is None else env)
    name = provider or env.get("COSTLESS_JUDGE_PROVIDER") or env.get("COSTLESS_PROVIDER")
    judge_env = {**env, "COSTLESS_PROVIDER": name or "anthropic"}
    if name != env.get("COSTLESS_PROVIDER"):
        # A different provider than the target's: don't inherit the target's model.
        judge_env.pop("COSTLESS_MODEL", None)
    chosen = model or env.get("COSTLESS_JUDGE_MODEL")
    if chosen:
        judge_env["COSTLESS_MODEL"] = chosen
    return provider_from_env(judge_env), model_from_env(judge_env)


def model_from_env(env: Mapping[str, str] | None = None) -> str:
    """The model to use: COSTLESS_MODEL, else the provider's default."""
    env = os.environ if env is None else env
    explicit = env.get("COSTLESS_MODEL", "").strip()
    if explicit:
        return explicit
    name = env.get("COSTLESS_PROVIDER", "anthropic").strip().lower()
    if name in DEFAULT_MODELS:
        return DEFAULT_MODELS[name]
    msg = f"COSTLESS_PROVIDER={name} has no default model; set COSTLESS_MODEL"
    raise ConfigError(msg)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"{key} must be an integer, got {raw!r}"
        raise ConfigError(msg) from exc
    if value < 0:
        msg = f"{key} must be >= 0"
        raise ConfigError(msg)
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"{key} must be a number, got {raw!r}"
        raise ConfigError(msg) from exc
    if value <= 0:
        msg = f"{key} must be > 0"
        raise ConfigError(msg)
    return value
