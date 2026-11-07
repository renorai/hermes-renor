"""How much time Renor's prefetch adds to a Hermes turn. Offline: no key, no network, no model.

    (cd engine && npm ci && npm run build)
    (cd integrations/hermes/tests && python3 bench_prefetch.py [--runs 200])

It measures three things and prints one JSON object:

- ``plugin_overhead_ms``: prefetch against an in-process Streamable HTTP server on 127.0.0.1 that answers at once,
  minus the same backend call made directly. That is the provider's own cost per turn (the thread, the deadline, the
  count), not the network's.
- ``loopback_prefetch_ms``: the whole prefetch against that server (HTTP + JSON-RPC + SSE parsing on this machine).
- ``local_engine_prefetch_ms``: the whole prefetch against the real open-source engine over stdio, on a vault that
  holds ``--notes`` saved memories.
- ``slow_server_prefetch_ms``: prefetch when Renor does not answer: the turn waits for the deadline, then goes on.

Hosted latency (mcp.renor.cloud) depends on the network between the user and Renor and is not measured here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

import test_renor_provider as base
from test_renor_provider import renor

QUERIES = ["What does Kestrel run on?", "Which branch do we deploy from?", "How should answers be formatted?",
           "What is the staging database?", "Which package manager does this repo use?"]


def summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"n": len(ordered), "p50": round(statistics.median(ordered), 2),
            "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 2), "max": round(ordered[-1], 2)}


def timed(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000


def provider_for(env: dict) -> "renor.RenorMemoryProvider":
    for key in base._ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    provider = renor.RenorMemoryProvider()
    assert provider.is_available(), provider.unavailable_reason()
    provider.initialize("bench", platform="cli", hermes_home=env["HERMES_HOME"], agent_context="primary")
    return provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--notes", type=int, default=50)
    args = parser.parse_args()
    home = tempfile.mkdtemp(prefix="renor-bench-home-")
    out: dict = {"runs": args.runs}

    base._FakeRenor.calls, base._FakeRenor.delay, base._FakeRenor.fail = [], {}, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), base._FakeRenor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    try:
        provider = provider_for({"HERMES_HOME": home, "RENOR_MCP_URL": url, "RENOR_API_KEY": base._FakeRenor.key})
        for q in QUERIES:  # warm: session open, code paths hot
            provider.prefetch(q)
        direct, through = [], []
        for i in range(args.runs):
            q = QUERIES[i % len(QUERIES)]
            direct.append(timed(lambda: provider._backend.recall(q)))
            through.append(timed(lambda: provider.prefetch(q)))
        out["loopback_direct_recall_ms"] = summary(direct)
        out["loopback_prefetch_ms"] = summary(through)
        out["plugin_overhead_ms"] = {"p50": round(statistics.median(through) - statistics.median(direct), 2)}

        base._FakeRenor.delay = {"get_context": 5}
        os.environ["RENOR_PREFETCH_TIMEOUT_MS"] = str(renor.DEFAULT_PREFETCH_TIMEOUT_MS)
        slow = provider_for({"HERMES_HOME": home, "RENOR_MCP_URL": url, "RENOR_API_KEY": base._FakeRenor.key})
        out["slow_server_prefetch_ms"] = {"timeout_ms": renor.DEFAULT_PREFETCH_TIMEOUT_MS,
                                          "observed": round(timed(lambda: slow.prefetch(QUERIES[0])), 1)}
        provider.shutdown()
    finally:
        base._FakeRenor.delay = {}
        server.shutdown()
        server.server_close()

    if base.ENGINE_CLI.exists() and shutil.which("node"):
        vault = tempfile.mkdtemp(prefix="renor-bench-vault-")
        try:
            local = provider_for({"HERMES_HOME": home, "RENOR_ENGINE_COMMAND": f"node {base.ENGINE_CLI} mcp",
                                  "RENOR_VAULT_PATH": vault})
            for i in range(args.notes):
                local.handle_tool_call("renor_remember", {"statement": f"Project {i} deploys from branch release-{i} "
                                                                       f"and uses Postgres {10 + i % 7}."})
            for q in QUERIES:
                local.prefetch(q)
            samples = [timed(lambda: local.prefetch(QUERIES[i % len(QUERIES)])) for i in range(max(20, args.runs // 4))]
            out["local_engine_prefetch_ms"] = {**summary(samples), "notes": args.notes,
                                               "timeouts": local.stats["prefetch_timeouts"]}
            local.shutdown()
        finally:
            shutil.rmtree(vault, ignore_errors=True)
    else:
        out["local_engine_prefetch_ms"] = "skipped: build the engine first (cd engine && npm run build)"
    shutil.rmtree(home, ignore_errors=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
