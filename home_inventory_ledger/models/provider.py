"""Shared Google GenAI SDK (`google-genai`) runtime for Gemini 4 & Decisions API.

Wraps:
- `client.interactions.create(...)` for `gemini-4-carbon` and `gemini-4-hydrogen`
  with multimodal inputs (images, audio WAV, uploaded video URIs) and JSON schema output.
- `client.decisions.create(...)` for `gemini-decision-1` closed-set `noul` and `choice`
  evaluations with calibrated probabilities and confidence scores.
- Clean JSON schema sanitization so Pydantic `$defs` / `$ref` schemas work reliably.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, TypeVar

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

# Automatically load workspace .env on import/reload so `uvicorn` always sees GEMINI_API_KEY
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

T = TypeVar("T", bound=BaseModel)


def _deref_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline `$defs` / `$ref` and strip unsupported meta-keys for strict API schema acceptance."""
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = str(node["$ref"])
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    return _walk(dict(defs[ref_name]))
            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k in {"$defs", "definitions", "title", "default"}:
                    continue
                cleaned[k] = _walk(v)
            return cleaned
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(schema)


def encode_file_b64(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        ext = p.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".wav": "audio/wav",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
        }.get(ext, "application/octet-stream")
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime


def image_part(path: str | Path, resolution: str | None = None) -> dict[str, Any]:
    b64, mime = encode_file_b64(path)
    part: dict[str, Any] = {"type": "image", "data": b64, "mime_type": mime}
    if resolution:
        part["resolution"] = resolution
    return part


def audio_part(path: str | Path) -> dict[str, Any]:
    b64, mime = encode_file_b64(path)
    return {"type": "audio", "data": b64, "mime_type": mime}


def extract_json_object(raw_text: str) -> str:
    """Strip markdown code fences if a model wraps JSON output."""
    txt = raw_text.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    return txt.strip()


class GenAIRuntime:
    """Thread-safe wrapper around `google.genai.Client` with telemetry counters and replay flag."""

    def __init__(
        self,
        api_key: str | None = None,
        carbon_model: str | None = None,
        hydrogen_model: str | None = None,
        decision_model: str | None = None,
        replay_mode: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.carbon_model = carbon_model or os.getenv("GEMINI_CARBON_MODEL", "gemini-4-carbon")
        self.hydrogen_model = hydrogen_model or os.getenv("GEMINI_HYDROGEN_MODEL", "gemini-4-hydrogen")
        self.decision_model = decision_model or os.getenv("GEMINI_DECISION_MODEL", "gemini-decision-1")

        if replay_mode is None:
            env_rep = os.getenv("LEDGER_REPLAY_MODE", "false").lower() in {"1", "true", "yes"}
            self.replay_mode = env_rep
        else:
            self.replay_mode = replay_mode

        self._lock = threading.Lock()
        self.carbon_calls = 0
        self.hydrogen_calls = 0
        self.decision_calls = 0
        self._client: genai.Client | None = None

    @property
    def has_live_credentials(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def client(self) -> genai.Client:
        if self._client is None:
            if not self.has_live_credentials:
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured. Set GEMINI_API_KEY in .env or environment, "
                    "or pass replay_mode=True for disclosed offline demo replay."
                )
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1beta"},
            )
        return self._client

    def record_call(self, role: str) -> None:
        with self._lock:
            if role == "carbon":
                self.carbon_calls += 1
            elif role == "hydrogen":
                self.hydrogen_calls += 1
            elif role == "decision":
                self.decision_calls += 1

    def upload_video_part(self, video_path: str | Path, timeout_s: float = 25.0) -> dict[str, Any] | None:
        """Upload video via `client.files.upload` and wait briefly until ACTIVE."""
        p = Path(video_path)
        if not p.exists():
            return None
        # For very small clips (< 8 MB), inline base64 or file upload both work; prefer Files API
        try:
            fobj = self.client().files.upload(file=str(p))
            t0 = time.time()
            while getattr(fobj, "state", None) and str(getattr(fobj.state, "name", fobj.state)).upper() == "PROCESSING":
                if time.time() - t0 > timeout_s:
                    break
                time.sleep(1.0)
                fobj = self.client().files.get(name=fobj.name)
            mime = getattr(fobj, "mime_type", None) or "video/mp4"
            return {"type": "video", "uri": fobj.uri, "mime_type": mime}
        except Exception:
            return None

    def structured_interaction(
        self,
        *,
        role: str,
        model: str,
        input_parts: list[dict[str, Any] | str],
        schema_cls: type[T],
        system_instruction: str | None = None,
        retries: int = 1,
    ) -> T:
        """Run `client.interactions.create` with Pydantic JSON schema validation + 1 retry."""
        clean_schema = _deref_json_schema(schema_cls.model_json_schema())
        normalized_parts: list[dict[str, Any]] = [
            {"type": "text", "text": p} if isinstance(p, str) else p for p in input_parts
        ]

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            self.record_call(role)
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": normalized_parts,
                    "response_mime_type": "application/json",
                    "response_format": clean_schema,
                }
                if system_instruction:
                    kwargs["system_instruction"] = system_instruction

                interaction = self.client().interactions.create(**kwargs)
                raw_out = getattr(interaction, "output_text", None) or ""
                if not raw_out and getattr(interaction, "steps", None):
                    # Extract text from interaction steps if output_text not top-level populated
                    raw_out = str(interaction.steps[-1])
                cleaned = extract_json_object(raw_out)
                return schema_cls.model_validate_json(cleaned)
            except Exception as err:
                last_err = err
                if attempt < retries:
                    # Append explicit schema correction hint on retry
                    normalized_parts = normalized_parts + [
                        {
                            "type": "text",
                            "text": (
                                f"Return strictly valid JSON matching this schema: "
                                f"{json.dumps(clean_schema)}. Previous validation error: {err}"
                            ),
                        }
                    ]
                    continue
        raise RuntimeError(f"{model} structured interaction failed: {last_err}") from last_err

    def evaluate_decisions(
        self,
        *,
        input_parts: list[dict[str, Any] | str] | str,
        questions: dict[str, dict[str, Any]],
    ) -> Any:
        """Call `client.decisions.create` on `gemini-decision-1`."""
        self.record_call("decision")
        if isinstance(input_parts, list):
            norm_inp: Any = [
                {"type": "text", "text": p} if isinstance(p, str) else p for p in input_parts
            ]
        else:
            norm_inp = input_parts

        return self.client().decisions.create(
            model=self.decision_model,
            input=norm_inp,
            questions=questions,
        )
