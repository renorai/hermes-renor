"""A minimal fake of the two hermes-agent modules the Renor provider imports (``agent.memory_provider`` and
``agent.secret_scope``), so the provider's own tests run without a Hermes checkout. It mirrors the interface of
NousResearch/hermes-agent v0.23.0 (2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19) and nothing else: no MemoryManager, no profiles,
no multiplexing. What needs the real Hermes is in ``test_hermes_end_to_end.py``.
"""
