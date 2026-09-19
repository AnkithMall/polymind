"""Web fetch skill — retrieve content from URLs."""

from __future__ import annotations

import urllib.error
import urllib.request

from polymind.core.skills.base import Skill
from polymind.core.skills.types import (
    Permission,
    SkillContext,
    SkillManifest,
    SkillParam,
    SkillResult,
)


class WebFetchSkill(Skill):
    """Fetch content from a URL."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="web_fetch",
            description=(
                "Fetch content from a URL. Returns the response text. "
                "Use for: reading documentation, checking APIs, "
                "retrieving public data. Only HTTP/HTTPS allowed."
            ),
            permissions=[Permission.NETWORK],
            params=[
                SkillParam(
                    name="url",
                    type="string",
                    description="URL to fetch (must be http:// or https://)",
                ),
                SkillParam(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds. Default: 15",
                    required=False,
                    default=15,
                ),
            ],
            examples=[
                'web_fetch(url="https://docs.python.org/3/library/json.html")',
                'web_fetch(url="https://api.github.com/repos/python/cpython")',
            ],
            tags=["network", "web", "fetch"],
            timeout_seconds=20,
        )

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        try:
            validated = self.validate_params(**kwargs)
            url = validated["url"]
            timeout = validated.get("timeout", 15) or 15

            if not url.startswith(("http://", "https://")):
                return SkillResult(success=False, error="Only HTTP/HTTPS URLs are allowed")

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Polymind/0.1 (local LLM toolkit)"},
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    content = resp.read(ctx.max_output_bytes + 1)
                    truncated = len(content) > ctx.max_output_bytes
                    text = content[: ctx.max_output_bytes].decode("utf-8", errors="replace")

                    return SkillResult(
                        success=True,
                        output=text,
                        truncated=truncated,
                        metadata={
                            "url": url,
                            "status": resp.status,
                            "content_type": resp.headers.get("Content-Type", ""),
                        },
                    )
            except urllib.error.HTTPError as e:
                return SkillResult(
                    success=False,
                    error=f"HTTP {e.code}: {e.reason}",
                    metadata={"url": url, "status": e.code},
                )
            except urllib.error.URLError as e:
                return SkillResult(
                    success=False,
                    error=f"URL error: {e.reason}",
                    metadata={"url": url},
                )
        except Exception as e:
            return SkillResult(success=False, error=str(e))
