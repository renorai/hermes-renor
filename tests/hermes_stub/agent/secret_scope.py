"""Fake of ``agent.secret_scope`` (hermes-agent v0.23.0): ``get_secret`` as a single-profile process reads it."""

from __future__ import annotations

import os
from typing import Optional


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)
