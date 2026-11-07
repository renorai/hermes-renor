"""Tests for the Renor memory provider, against a stub of the two hermes-agent modules it imports.

Run from the repository root, with a built engine (``cd engine && npm ci && npm run build``); no Hermes checkout:

    (cd integrations/hermes/tests && python3 -m unittest -v test_renor_provider test_review_874 test_lifecycle)

The provider is loaded against ``hermes_stub/`` (a fake of ``agent.memory_provider`` and ``agent.secret_scope``) and
driven directly, as Hermes's MemoryManager drives it. What needs the real MemoryManager is in
``test_hermes_end_to_end.py``, which runs only against a hermes-agent checkout (``integrations/hermes/README.md``).

The local-engine tests start the real open-source engine (``engine/dist/cli.js mcp``) offline on a temporary
folder, and fail rather than skip when it is not built. The hosted tests run against a small in-process MCP server
that speaks Streamable HTTP, so they check the wire calls without a network or a key.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
# The built open-source engine: this repository's engine/, or RENOR_ENGINE_CLI outside it (the standalone plugin repo).
ENGINE_CLI = Path(os.environ.get("RENOR_ENGINE_CLI") or PLUGIN_DIR.parents[1] / "engine" / "dist" / "cli.js")
STUB_DIR = Path(__file__).resolve().parent / "hermes_stub"


def load_provider(module_name: str, agent_path: Path | None):
    """The plugin module, executed against the ``agent`` package at ``agent_path`` (the stub when None).

    ``agent`` is swapped into ``sys.modules`` only while the module executes and put back after, so the stubbed tests
    and the end-to-end tests can share one process without either seeing the other's ``agent``."""
    ours = lambda name: name == "agent" or name.startswith("agent.")  # noqa: E731
    saved = {name: sys.modules.pop(name) for name in [n for n in sys.modules if ours(n)]}
    sys.path.insert(0, str(agent_path or STUB_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(agent_path or STUB_DIR))
        for name in [n for n in sys.modules if ours(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


renor = load_provider("renor_hermes_stubbed", None)

_ENV_KEYS = ("HERMES_HOME", "RENOR_API_KEY", "RENOR_MCP_URL", "RENOR_ENGINE_COMMAND", "RENOR_VAULT_PATH", "RENOR_SCOPE",
             "RENOR_AUTO_SAVE", "RENOR_PREFETCH_TIMEOUT_MS", "RENOR_ALLOW_SHARING")


class _EnvCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
        self.home = tempfile.mkdtemp(prefix="renor-hermes-home-")
        os.environ["HERMES_HOME"] = self.home  # the active profile, so Hermes lets calls reach the provider

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        shutil.rmtree(self.home, ignore_errors=True)

    def start(self, agent_context: str = "primary"):
        """The provider, initialized as Hermes initializes it (``initialize_all`` passes these keywords)."""
        provider = renor.RenorMemoryProvider()
        self.assertTrue(provider.is_available(), provider.unavailable_reason())
        provider.initialize("s1", platform="cli", hermes_home=self.home, agent_context=agent_context)
        self.addCleanup(provider.shutdown)
        return provider

    @staticmethod
    def tool(provider, name: str, args: dict) -> dict:
        return json.loads(provider.handle_tool_call(name, args))


class LocalEngineTest(_EnvCase):
    """The self-hosted engine, started as `renor-engine mcp` over stdio, offline."""

    def setUp(self) -> None:
        # Fail, not skip: a runner without the built engine would otherwise report these as passing.
        self.assertTrue(ENGINE_CLI.exists() and shutil.which("node"), "needs node and a built engine: cd engine && npm run build")
        super().setUp()
        self.vault = tempfile.mkdtemp(prefix="renor-hermes-vault-")
        self.addCleanup(shutil.rmtree, self.vault, True)
        os.environ["RENOR_ENGINE_COMMAND"] = f"node {ENGINE_CLI} mcp"
        os.environ["RENOR_VAULT_PATH"] = self.vault

    def test_remember_then_recall_through_tools_and_prefetch(self) -> None:
        provider = self.start()
        self.assertEqual({s["name"] for s in provider.get_tool_schemas()}, {"renor_recall", "renor_remember"})

        saved = self.tool(provider, "renor_remember", {
            "statement": "Kestrel deploys to fly.io from the main branch",
            "subject": "Kestrel", "predicate": "database", "object": "Postgres 16"})
        self.assertTrue(saved["saved"], saved)
        self.assertTrue(saved["receipt"]["fact"]["ok"], saved)
        self.assertTrue(saved["receipt"]["note"]["ok"], saved)
        self.assertTrue((Path(self.vault) / "Memory" / "Daily").is_dir())

        found = self.tool(provider, "renor_recall", {"query": "Which database does Kestrel use?"})["memories"]
        self.assertTrue(any("Postgres 16" in m for m in found), found)
        self.assertTrue(any("fly.io" in m for m in found), found)

        block = provider.prefetch("where does Kestrel deploy to?")
        self.assertRegex(block, r"Renor holds \d+ saved memor")
        self.assertNotIn("fly.io", block)  # automatic recall carries a count, never the stored text
        self.assertEqual(provider.recall_status().provider_label, "Renor")
        self.assertGreater(provider.recall_status().count, 0)

        self.assertEqual(provider.prefetch("thanks!"), "")  # a bare acknowledgement recalls nothing
        self.assertIsNone(provider.recall_status())  # and the indicator reflects only that last prefetch

    def test_refusals_are_returned_not_raised(self) -> None:
        provider = self.start()
        self.assertIn("all three or none", self.tool(provider, "renor_remember", {"statement": "x", "subject": "Kestrel"})["error"])
        self.assertIn("required", self.tool(provider, "renor_recall", {"query": " "})["error"])

    def test_builtin_memory_writes_are_mirrored_by_the_primary_agent_only(self) -> None:
        provider = self.start()
        provider.on_memory_write("add", "user", "Prefers answers in Brazilian Portuguese")
        provider.on_memory_write("remove", "user", "Something the user withdrew")
        self.assertTrue(provider.flush(20))
        found = self.tool(provider, "renor_recall", {"query": "Brazilian Portuguese"})["memories"]
        self.assertTrue(any("Brazilian Portuguese" in m for m in found), found)
        self.assertFalse(any("withdrew" in m for m in self.tool(provider, "renor_recall", {"query": "withdrew"})["memories"]))

        sub_provider = self.start(agent_context="subagent")
        sub_provider.on_memory_write("add", "memory", "A subagent scratch observation")
        self.assertEqual(sub_provider.stats["queued"], 0)

    def test_shutdown_stops_the_engine(self) -> None:
        provider = self.start()
        self.tool(provider, "renor_recall", {"query": "anything at all"})
        process = provider._client._transport._process
        provider.shutdown()
        self.assertIsNotNone(process.poll())


# The real get_context shapes (engine/src/mvp/agentMemory.ts): a fact has no `text`, a memory does.
_DEFAULT_FACTS = [{"subject": "Kestrel", "predicate": "database", "object": "Postgres 16",
                   "validFrom": "2026-10-21T10:00:00.000Z", "scope": "global"}]
_DEFAULT_MEMORIES = [{"notePath": "Memory/Daily/2026-10-21.md", "text": "Kestrel deploys to fly.io", "reason": "fts"},
                     {"notePath": "Memory/Daily/2026-10-20.md", "text": "unrelated", "reason": "fts", "matched": False}]


class _FakeRenor(BaseHTTPRequestHandler):
    """Streamable HTTP MCP: a session id on initialize, SSE replies to tools/call, 401 without the key."""

    calls: list = []
    key = "rk_test"
    facts: list = _DEFAULT_FACTS
    memories: list = _DEFAULT_MEMORIES
    relevance: dict | None = None
    redirect_to = ""  # when set, every POST is answered with a 302 to this URL
    delay: dict = {}  # tool name -> seconds to wait before answering it
    fail: list = []  # HTTP statuses answered to the next `remember` calls, one each, before they succeed

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.redirect_to:
            return self._send(302, "application/json", "", {"Location": self.redirect_to})
        if self.headers.get("Authorization") != f"Bearer {self.key}":
            return self._send(401, "application/json", json.dumps({"error": "invalid_token"}))
        if message.get("method") == "initialize":
            body = {"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": renor.PROTOCOL_VERSION, "capabilities": {"tools": {}}}}
            return self._send(200, "application/json", json.dumps(body), {"Mcp-Session-Id": "sess-1"})
        if "id" not in message:
            return self._send(202, "application/json", "")
        if self.headers.get("Mcp-Session-Id") != "sess-1":
            return self._send(404, "application/json", json.dumps({"error": "Session not found"}))
        params = message["params"]
        time.sleep(self.delay.get(params["name"], 0))
        if params["name"] == "remember" and type(self).fail:
            status = type(self).fail.pop(0)
            return self._send(status, "application/json", json.dumps({"error": "unavailable"}))
        type(self).calls.append((params["name"], params["arguments"]))
        if params["name"] == "get_context":
            result = {"ok": True, "facts": self.facts, "memories": self.memories,
                      **({"relevance": self.relevance} if self.relevance else {})}
        else:
            result = {"ok": True, "action": "store"}
        body = {"jsonrpc": "2.0", "id": message["id"], "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}
        self._send(200, "text/event-stream", f"event: message\ndata: {json.dumps(body)}\n\n")

    def _send(self, status: int, kind: str, body: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())


class HostedTest(_EnvCase):
    def setUp(self) -> None:
        super().setUp()
        _FakeRenor.calls = []
        _FakeRenor.facts, _FakeRenor.memories, _FakeRenor.relevance = _DEFAULT_FACTS, _DEFAULT_MEMORIES, None
        _FakeRenor.redirect_to = ""
        _FakeRenor.delay, _FakeRenor.fail = {}, []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeRenor)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        os.environ["RENOR_MCP_URL"] = f"http://127.0.0.1:{self.server.server_port}/mcp"

    def test_not_available_without_a_key(self) -> None:
        provider = renor.RenorMemoryProvider()
        self.assertFalse(provider.is_available())
        self.assertIn("RENOR_API_KEY", provider.unavailable_reason())

    def test_recall_and_remember_use_get_context_and_remember(self) -> None:
        os.environ["RENOR_API_KEY"] = _FakeRenor.key
        os.environ["RENOR_SCOPE"] = "project:kestrel"
        provider = self.start()

        block = provider.prefetch("What does Kestrel run on?")
        self.assertIn("Renor holds 2 saved memories", block)  # the fact and the matched memory; not the unmatched one
        found = self.tool(provider, "renor_recall", {"query": "What does Kestrel run on?"})["memories"]
        self.assertEqual(found, ["Kestrel database: Postgres 16", "Kestrel deploys to fly.io"])

        saved = self.tool(provider, "renor_remember", {"statement": "Kestrel uses Postgres 16",
                                                       "subject": "Kestrel", "predicate": "database", "object": "Postgres 16"})
        self.assertTrue(saved["saved"], saved)
        self.assertEqual(_FakeRenor.calls, [
            ("get_context", {"query": "What does Kestrel run on?", "max_tokens": 800, "scope": "project:kestrel"}),
            ("get_context", {"query": "What does Kestrel run on?", "max_tokens": 800, "scope": "project:kestrel"}),
            ("remember", {"statement": "Kestrel uses Postgres 16", "source": "hermes", "subject": "Kestrel",
                          "predicate": "database", "object": "Postgres 16", "scope": "project:kestrel"}),
        ])

    def test_a_wrong_key_is_reported_by_the_tool(self) -> None:
        os.environ["RENOR_API_KEY"] = "wrong"
        provider = self.start()
        self.assertEqual(provider.prefetch("What does Kestrel run on?"), "")
        error = self.tool(provider, "renor_recall", {"query": "Kestrel"})["error"]
        self.assertIn("HTTP 401", error)
        self.assertIn("RENOR_API_KEY", error)

    def test_an_expired_session_is_reopened_once(self) -> None:
        os.environ["RENOR_API_KEY"] = _FakeRenor.key
        provider = self.start()
        self.tool(provider, "renor_recall", {"query": "Kestrel"})
        provider._client._transport.session_id = "stale"
        self.assertIn("memories", self.tool(provider, "renor_recall", {"query": "Kestrel"}))
        self.assertEqual(len(_FakeRenor.calls), 2)


if __name__ == "__main__":
    unittest.main()
