"""HTTP, on the standard library.

Every vendor here speaks JSON over HTTPS, and their SDKs bring dependency
trees far larger than the twenty lines they would save. `urllib` is enough, and
it keeps the project installable with one `pip install` line.

The transport is a seam so tests can replace it. That is the whole reason the
real policy is testable without a network: there is exactly one place where
bytes leave the process, and it is injectable.

One honest limitation: `urlopen` is blocking, so it runs in a worker thread.
Cancelling the decision (which the deadline arbiter does routinely) unblocks
the caller immediately but cannot kill that thread -- it lives until the socket
timeout. This is why `timeout_s` is bounded and configured close to the
decision deadline rather than left at the system default.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Protocol

MAX_ERROR_BODY_CHARS = 400


class VLMTransportError(RuntimeError):
    """Network or HTTP-level failure. Treated by the policy as a failed decision."""


class HttpTransport(Protocol):
    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]: ...


class UrllibTransport:
    name = "urllib"

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(self._post, url, dict(headers), dict(payload), timeout_s)

    def _post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float
    ) -> Mapping[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The response body explains the failure; the request headers carry
            # the credential, so they are never included here.
            detail = exc.read().decode("utf-8", "replace")[:MAX_ERROR_BODY_CHARS]
            raise VLMTransportError(f"HTTP {exc.code} from the model API: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VLMTransportError(f"could not reach the model API: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise VLMTransportError("the model API returned a body that is not JSON") from exc
