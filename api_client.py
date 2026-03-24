from __future__ import annotations

import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


class VLMClientError(RuntimeError):
    """Raised when the VLM service request fails or returns an error."""


@dataclass
class GenerateParams:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    def to_form(self) -> Dict[str, str]:
        return {
            "max_new_tokens": str(self.max_new_tokens),
            "do_sample": str(self.do_sample).lower(),
            "temperature": str(self.temperature),
            "top_p": str(self.top_p),
            "top_k": str(self.top_k),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


@dataclass
class GenerateResult:
    ok: bool
    text: str
    n_images: int
    elapsed_sec: float
    status_code: int
    raw: Dict[str, Any] = field(default_factory=dict)


class QwenVLMClient:
    """Lightweight client for the existing Qwen3-VL FastAPI service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8002",
        timeout: int = 180,
        retries: int = 2,
        retry_wait_sec: float = 1.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_wait_sec = retry_wait_sec
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_from_files(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        """Use multipart upload. This is the safest mode for local Windows/WSL usage."""
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        params = params or GenerateParams()
        url = f"{self.base_url}/v1/vlm/generate"
        data = {"prompt": prompt, **params.to_form()}

        opened_files = []
        files = []
        try:
            for img_path in image_paths:
                path = Path(img_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Image not found: {path}")
                fp = path.open("rb")
                opened_files.append(fp)
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("images", (path.name, fp, mime)))

            return self._request_with_retry("post", url, data=data, files=files)
        finally:
            for fp in opened_files:
                try:
                    fp.close()
                except Exception:
                    pass

    def generate_from_image_base64(
        self,
        prompt: str,
        image_base64_list: Sequence[str],
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        if not image_base64_list:
            raise ValueError("image_base64_list must not be empty")

        params = params or GenerateParams()
        url = f"{self.base_url}/v1/vlm/generate_json"
        payload = {
            "prompt": prompt,
            "image_base64_list": list(image_base64_list),
            **params.to_json(),
        }
        return self._request_with_retry("post", url, json=payload)

    def generate_from_container_paths(
        self,
        prompt: str,
        image_paths: Sequence[str],
        params: Optional[GenerateParams] = None,
    ) -> GenerateResult:
        """Use this only when the given paths are visible inside the container."""
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        params = params or GenerateParams()
        url = f"{self.base_url}/v1/vlm/generate_json"
        payload = {
            "prompt": prompt,
            "image_paths": list(image_paths),
            **params.to_json(),
        }
        return self._request_with_retry("post", url, json=payload)

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> GenerateResult:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                elapsed = time.perf_counter() - start
                return self._parse_generate_response(resp, elapsed)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_wait_sec)

        raise VLMClientError(f"Request failed after retries: {last_err}") from last_err

    @staticmethod
    def _parse_generate_response(resp: requests.Response, elapsed: float) -> GenerateResult:
        status_code = resp.status_code
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise VLMClientError(
                f"Non-JSON response. status={status_code}, body={resp.text[:500]}"
            ) from exc

        if status_code >= 400:
            raise VLMClientError(f"HTTP {status_code}: {payload}")

        if isinstance(payload, dict) and payload.get("error"):
            raise VLMClientError(str(payload["error"]))

        return GenerateResult(
            ok=True,
            text=str(payload.get("text", "")),
            n_images=int(payload.get("n_images", 0)),
            elapsed_sec=elapsed,
            status_code=status_code,
            raw=payload,
        )


if __name__ == "__main__":
    client = QwenVLMClient()
    info = client.health()
    print(json.dumps(info, ensure_ascii=False, indent=2))
