"""Fake of ``agent.memory_provider`` (hermes-agent v0.23.0, 2bd378ed): the names the Renor provider imports."""

from __future__ import annotations

import contextvars
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


def spawn_context_thread(target: Callable[..., Any], *, name: str, daemon: bool = True,
                         args: tuple = (), kwargs: Optional[dict[str, Any]] = None) -> threading.Thread:
    """An unstarted thread running ``target`` under the caller's contextvars, as Hermes returns it."""
    ctx = contextvars.copy_context()
    return threading.Thread(target=lambda *a, **k: ctx.run(target, *a, **k), args=args, kwargs=kwargs, name=name,
                            daemon=daemon)


@dataclass(frozen=True)
class RecallStatus:
    provider_label: str
    count: int
    glyph: str = "🧠"


# Hermes's own definition (agent/memory_provider.py), copied so a trivial prompt means the same thing here.
TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'hi|hey|hello|yo|sup|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"' + "'" + r'~‘’“”—–…()\[\]{}<>*&^%$#@!+=` ]*$',
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    """The abstract methods and the defaults the Renor provider relies on."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    def unavailable_reason(self) -> str:
        return ""

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def recall_status(self) -> Optional[RecallStatus]:
        return None

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[list[dict[str, Any]]] = None, turn_author: Optional[dict[str, Any]] = None) -> None:
        pass

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        pass

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        pass

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return ""

    def get_config_schema(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass
