"""The provider's per-turn lifecycle, against the hermes-agent stub: bounded recall before each turn (prefetch),
saving after it (sync_turn), before compaction (on_pre_compress) and at the end (on_session_end, shutdown), the
writer's queue and retries, and what leaves the machine (sharing off for the local engine, no secrets saved or logged).

Run with the other stubbed tests (a built engine is needed for the local-engine case):

    (cd integrations/hermes/tests && python3 -W error::ResourceWarning -m unittest -v test_renor_provider test_review_874 test_lifecycle)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import test_renor_provider as base
from test_renor_provider import renor

KEY_WITNESS = "rk_test"


def remembered() -> list[str]:
    return [args["statement"] for name, args in base._FakeRenor.calls if name == "remember"]


class DurableStatementsTest(unittest.TestCase):
    """What sync_turn picks out of a user message, by local rules only."""

    KEPT = [
        "Remember that I deploy Kestrel to fly.io.",
        "From now on, answer in Brazilian Portuguese.",
        "Always use pnpm in this repo.",
        "I prefer short answers with code first.",
        "We decided to keep Postgres 16 for Kestrel.",
        "My timezone is America/Sao_Paulo.",
        "Our main branch is trunk.",
        "for the record, the staging box runs Debian 12",
        "ok so we use Vitest, not Jest",
        "Always put new panels under src/components/dashboard/settings/learning/panels.",
    ]
    DROPPED = [
        "Can you fix the failing test?",
        "Remember to run the tests before you push.",
        "Do you remember what we decided?",
        "Fix the bug in parser.ts",
        "I run npm test now and then push.",
        "I like that, thanks",
        "never mind",
        "My API key is sk-live-abcdefghijklmnop1234",
        "Remember that the token is ghp_abcdefghijklmnopqrstuvwxyz0123",
        "Remember that the deploy hook is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "yes",
        "```\nalways run this\n```",
    ]

    def test_durable_statements_are_kept(self) -> None:
        for text in self.KEPT:
            with self.subTest(text=text):
                self.assertEqual(renor.durable_statements(text), [text.strip()])

    def test_questions_tasks_code_and_secrets_are_dropped(self) -> None:
        for text in self.DROPPED:
            with self.subTest(text=text):
                self.assertEqual(renor.durable_statements(text), [])

    def test_at_most_three_per_message_from_the_middle_of_a_paragraph(self) -> None:
        text = ("Here is the plan for today. I prefer tabs. Fix the CI job first. Always squash merge. "
                "We use Bun. My editor is Helix.")
        self.assertEqual(renor.durable_statements(text), ["I prefer tabs.", "Always squash merge.", "We use Bun."])


class WriterTest(unittest.TestCase):
    """The bounded queue and retries, without a server."""

    def test_a_full_queue_drops_new_items_instead_of_blocking(self) -> None:
        gate = threading.Event()
        writer = renor._Writer(lambda s, t: gate.wait(5), size=2)
        started = time.monotonic()
        results = [writer.submit(f"item {i}") for i in range(5)]
        self.assertLess(time.monotonic() - started, 0.5)
        # One is taken by the writer thread, two wait in the queue; the rest are dropped.
        self.assertGreaterEqual(writer.dropped, 2)
        self.assertEqual(results.count(False), writer.dropped)
        gate.set()
        self.assertTrue(writer.close(5))

    def test_transient_failures_are_retried_and_refusals_are_not(self) -> None:
        attempts: list[str] = []

        def save(statement, triple):
            attempts.append(statement)
            if statement == "flaky" and attempts.count("flaky") < 3:
                raise renor.RenorError("Renor answered HTTP 503: busy")
            if statement == "refused":
                raise renor.RenorError("Renor answered HTTP 400: bad statement")

        writer = renor._Writer(save, backoff=(0.01, 0.02))
        writer.submit("flaky")
        writer.submit("refused")
        self.assertTrue(writer.flush(5))
        self.assertEqual((attempts.count("flaky"), attempts.count("refused")), (3, 1))
        self.assertEqual((writer.saved, writer.failed, writer.retried), (1, 1, 2))
        writer.close(1)

    def test_flush_is_bounded(self) -> None:
        writer = renor._Writer(lambda s, t: time.sleep(2))
        writer.submit("slow")
        started = time.monotonic()
        self.assertFalse(writer.flush(0.2))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(writer.close(5))


class HostedLifecycleTest(base.HostedTest):
    # HostedTest's own tests run in test_renor_provider; here only the lifecycle ones.
    locals().update({n: None for n in vars(base.HostedTest) if n.startswith("test_")})

    def setUp(self) -> None:
        super().setUp()
        os.environ["RENOR_API_KEY"] = base._FakeRenor.key

    def test_sync_turn_saves_the_durable_statements_in_the_background(self) -> None:
        provider = self.start()
        base._FakeRenor.delay = {"remember": 0.5}
        started = time.monotonic()
        provider.sync_turn("Fix the flaky test. From now on, run pnpm test before every commit.", "Done.")
        self.assertLess(time.monotonic() - started, 0.1, "sync_turn must not wait for the network")
        provider.sync_turn("thanks!", "You're welcome.")
        provider.sync_turn("What does Kestrel run on?", "Postgres 16.")
        self.assertTrue(provider.flush(10))
        self.assertEqual(remembered(), ["From now on, run pnpm test before every commit."])
        saved = [args for name, args in base._FakeRenor.calls if name == "remember"][0]
        self.assertEqual(saved["source"], "hermes")

    def test_transient_failures_are_retried_until_saved(self) -> None:
        provider = self.start()
        base._FakeRenor.fail = [503, 502]
        provider.sync_turn("I prefer answers with the command first.", "Noted.")
        self.assertTrue(provider.flush(15))
        self.assertEqual(remembered(), ["I prefer answers with the command first."])
        self.assertEqual((provider._writer.saved, provider._writer.retried), (1, 2))

    def test_turns_mode_saves_each_turn_and_off_saves_nothing(self) -> None:
        os.environ["RENOR_AUTO_SAVE"] = "turns"
        provider = self.start()
        provider.sync_turn("Rename the module to ledger.", "Renamed it in 4 files.")
        provider.sync_turn("ok", "…")
        self.assertTrue(provider.flush(10))
        self.assertEqual(len(remembered()), 1)
        self.assertIn("User: Rename the module to ledger.", remembered()[0])
        self.assertIn("Assistant: Renamed it in 4 files.", remembered()[0])

        base._FakeRenor.calls = []
        os.environ["RENOR_AUTO_SAVE"] = "off"
        quiet = self.start()
        quiet.sync_turn("From now on, use tabs.", "OK.")
        quiet.on_session_end([{"role": "user", "content": "Always use tabs."}])
        quiet.on_memory_write("add", "user", "Prefers tabs")  # an explicit write in Hermes is still mirrored
        self.assertTrue(quiet.flush(10))
        self.assertEqual(remembered(), ["Prefers tabs"])
        self.assertNotIn("saved to Renor after each turn", quiet.system_prompt_block())

    def test_subagents_and_cron_runs_never_write_automatically(self) -> None:
        for context in ("subagent", "cron", "flush"):
            provider = self.start(agent_context=context)
            provider.sync_turn("From now on, use tabs.", "OK.")
            provider.on_memory_write("add", "memory", "A scratch observation")
            self.assertEqual(provider.on_pre_compress([{"role": "user", "content": "Always use tabs."}]), "")
            provider.on_session_end([{"role": "user", "content": "Always use tabs."}])
        self.assertEqual(remembered(), [])

    def test_pre_compress_saves_what_the_turns_did_not_and_nothing_twice(self) -> None:
        provider = self.start()
        provider.sync_turn("We deploy Kestrel from the release branch.", "OK.")
        span = [
            {"role": "system", "content": "Always obey the system prompt."},
            {"role": "user", "content": "We deploy Kestrel from the release branch."},
            {"role": "assistant", "content": "I prefer to check first."},
            {"role": "user", "content": [{"type": "text", "text": "My editor is Helix. Open main.rs."}]},
            {"role": "tool", "content": "Always run as root."},
        ]
        note = provider.on_pre_compress(span)
        self.assertIn("Renor saved 1 durable statement", note)
        self.assertEqual(provider._writer.pending(), 0, "on_pre_compress waits for the queue to drain")
        self.assertEqual(remembered(), ["We deploy Kestrel from the release branch.", "My editor is Helix."])
        self.assertEqual(provider.on_pre_compress(span), "")
        provider.on_session_end(span)
        self.assertEqual(len(remembered()), 2)

    def test_a_statement_saved_with_the_tool_is_not_saved_again(self) -> None:
        provider = self.start()
        self.tool(provider, "renor_remember", {"statement": "I prefer tabs."})
        provider.sync_turn("I prefer tabs.", "Saved.")
        self.assertTrue(provider.flush(10))
        self.assertEqual(remembered(), ["I prefer tabs."])

    def test_shutdown_flushes_what_is_queued(self) -> None:
        provider = renor.RenorMemoryProvider()
        provider.initialize("s1", platform="cli", hermes_home=self.home, agent_context="primary")
        base._FakeRenor.delay = {"remember": 0.3}
        provider.sync_turn("Always sign commits.", "OK.")
        provider.shutdown()
        self.assertEqual(remembered(), ["Always sign commits."])

    def test_prefetch_is_bounded_and_skipped_when_renor_is_slow(self) -> None:
        os.environ["RENOR_PREFETCH_TIMEOUT_MS"] = "200"
        provider = self.start()
        self.assertTrue(provider.prefetch("What does Kestrel run on?"))  # fast: answered
        base._FakeRenor.delay = {"get_context": 1.5}
        started = time.monotonic()
        self.assertEqual(provider.prefetch("What does Kestrel run on?"), "")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIsNone(provider.recall_status())
        self.assertEqual(provider.prefetch("Which branch do we deploy from?"), "")  # the slow one is still out
        self.assertEqual((provider.stats["prefetch_timeouts"], provider.stats["prefetch_busy"]), (1, 1))
        base._FakeRenor.delay = {}
        provider._prefetch_thread.join(5)
        self.assertIn("Renor holds", provider.prefetch("What does Kestrel run on?"))

    def test_the_timeout_setting_is_clamped(self) -> None:
        for raw, seconds in (("50", 0.1), ("99999", 10.0), ("abc", 1.5), ("", 1.5), ("800", 0.8)):
            os.environ["RENOR_PREFETCH_TIMEOUT_MS"] = raw
            self.assertEqual(self.start()._prefetch_timeout, seconds, raw)

    def test_a_secret_is_never_saved_and_the_key_is_never_logged(self) -> None:
        os.environ["RENOR_AUTO_SAVE"] = "turns"
        provider = self.start()
        with self.assertLogs(renor.logger, level=logging.DEBUG) as logs:
            renor.logger.debug("capture starts")
            provider.sync_turn("Use this key: sk-live-abcdefghijklmnop1234 for the deploy.", "OK.")
            base._FakeRenor.fail = [401, 401, 401]
            provider.sync_turn("Rename the module to ledger.", "Done.")
            self.assertTrue(provider.flush(10))
            base._FakeRenor.delay = {"get_context": 2}
            os.environ["RENOR_API_KEY"] = "wrong"
            provider.prefetch("What does Kestrel run on?")
        self.assertEqual(remembered(), [])
        self.assertEqual(provider._writer.failed, 1, "a 401 is not retried")
        self.assertFalse(any(KEY_WITNESS in line for line in logs.output), logs.output)


class LocalSharingTest(base._EnvCase):
    """The engine's own default shares anonymous usage numbers; started from Hermes it shares nothing unless asked."""

    def child_env(self) -> dict:
        captured: dict = {}

        def popen(*args, **kwargs):
            captured.update(kwargs["env"])
            raise OSError("not started in this test")

        os.environ["RENOR_ENGINE_COMMAND"] = f"{shutil.which('python3')} -c pass"
        with mock.patch.object(renor.subprocess, "Popen", popen):
            provider = self.start()
            provider._client.warm()
        return captured

    def test_sharing_is_off_unless_the_user_opts_in(self) -> None:
        os.environ["RENOR_API_KEY"] = "rk_must_not_reach_the_child"
        env = self.child_env()
        self.assertEqual((env.get("DO_NOT_TRACK"), env.get("RENOR_SHARING")), ("1", "off"))
        self.assertNotIn("RENOR_API_KEY", env)
        os.environ["RENOR_ALLOW_SHARING"] = "1"
        env = self.child_env()
        self.assertNotIn("DO_NOT_TRACK", env)
        self.assertNotIn("RENOR_SHARING", env)


class LocalEngineLifecycleTest(base._EnvCase):
    """The real open-source engine, offline: a statement said in one turn is recalled in the next session."""

    def setUp(self) -> None:
        self.assertTrue(base.ENGINE_CLI.exists() and shutil.which("node"), "needs node and a built engine: cd engine && npm run build")
        super().setUp()
        self.vault = tempfile.mkdtemp(prefix="renor-hermes-vault-")
        self.addCleanup(shutil.rmtree, self.vault, True)
        os.environ["RENOR_ENGINE_COMMAND"] = f"node {base.ENGINE_CLI} mcp"
        os.environ["RENOR_VAULT_PATH"] = self.vault

    def test_said_once_recalled_in_the_next_session(self) -> None:
        first = self.start()
        first.sync_turn("Remember that Kestrel's staging database is Postgres 16 on Neon.", "Saved.")
        first.on_session_end([])
        first.shutdown()

        second = self.start()
        self.assertRegex(second.prefetch("Which database does Kestrel staging use?"), r"Renor holds \d+ saved memor")
        found = self.tool(second, "renor_recall", {"query": "Kestrel staging database"})["memories"]
        self.assertTrue(any("Neon" in m for m in found), found)


if __name__ == "__main__":
    unittest.main()
