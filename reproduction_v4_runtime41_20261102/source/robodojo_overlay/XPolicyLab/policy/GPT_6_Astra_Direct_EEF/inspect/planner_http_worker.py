"""Execute one planner HTTP request outside the Isaac Sim process."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import (
    CapabilityFailure,
    InfrastructureFailure,
    classify_openai_error,
    responses_base_url,
)


def _response_dict(response: Any) -> dict[str, Any]:
    try:
        return json.loads(response.model_dump_json())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        raise InfrastructureFailure("provider returned malformed JSON") from None


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    key_env = str(payload["key_env"])
    api_key = str(os.environ.get(key_env, "")).strip()
    if not api_key:
        raise InfrastructureFailure(
            f"API key environment variable {key_env!r} is unset or empty"
        )

    request = dict(payload["request"])
    timeout_s = float(payload["timeout_s"])
    api_style = str(payload["api_style"])
    if api_style == "responses":
        from openai import OpenAI
        from pathlib import Path
        import openai
        import typing_extensions
        prefix = Path(sys.prefix).resolve()
        if not all(Path(module.__file__).resolve().is_relative_to(prefix)
                   for module in (openai, typing_extensions)):
            raise InfrastructureFailure("HTTP worker imported SDK dependencies outside its own environment")

        client = OpenAI(
            api_key=api_key,
            base_url=responses_base_url(str(payload["azure_endpoint"])),
            max_retries=0,
            timeout=timeout_s,
        )
        journal = None
        if "journal" in payload:
            from kyb.trace import HttpJournal, write_once
            import importlib.metadata
            info = payload["journal"]
            journal = HttpJournal(info["root"], role=info["role"])
            write_once(journal.root / "model" / info["attempt_id"] / "worker.json", {
                "executable": sys.executable, "openai": __import__("openai").__file__,
                "typing_extensions": __import__("typing_extensions").__file__,
                "openai_version": importlib.metadata.version("openai"),
                "isaac_imported": any(n.startswith("isaacsim") for n in sys.modules),
            })
            journal.dispatched(info["attempt_id"])
        try:
            response = _response_dict(client.responses.create(**request))
        except Exception as error:
            if journal is not None:
                journal.failed(info["attempt_id"], status=getattr(error, "status_code", None),
                               error_type=type(error).__name__)
            raise
        if journal is not None:
            journal.complete(info["attempt_id"], response)
        return response
    if api_style == "chat":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=str(payload["azure_endpoint"]),
            api_key=api_key,
            api_version=str(payload["api_version"]),
            max_retries=0,
            timeout=timeout_s,
        )
        return _response_dict(client.chat.completions.create(**request))
    raise InfrastructureFailure(f"unsupported planner API style {api_style!r}")


def _failure_payload(
    failure: CapabilityFailure | InfrastructureFailure,
) -> dict[str, Any]:
    if isinstance(failure, CapabilityFailure):
        return {"kind": "capability", "message": str(failure)}
    return {
        "kind": "infrastructure",
        "message": str(failure),
        "retryable": failure.retryable,
        "retry_after_s": failure.retry_after_s,
        "status": failure.status,
        "key_unusable": failure.key_unusable,
    }


def main() -> None:
    payload = {}
    try:
        payload = json.load(sys.stdin)
        response = _execute(payload)
        envelope = {"response": response}
    except (CapabilityFailure, InfrastructureFailure) as failure:
        envelope = {"failure": _failure_payload(failure)}
    except Exception as error:
        envelope = {"failure": _failure_payload(classify_openai_error(error))}
    if "failure" in envelope and payload.get("journal"):
        from kyb.trace import HttpJournal
        info = payload["journal"]
        journal = HttpJournal(info["root"], role=info["role"])
        if journal.last_state(info["attempt_id"]) == "prepared":
            journal.state(info["attempt_id"], "failed_before_dispatch",
                          error_type="WorkerConfigurationFailure", usage=None)
    json.dump(envelope, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
