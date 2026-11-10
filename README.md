# Renor memory provider for Hermes Agent

**The problem.** Every agent you use starts from zero. You tell Claude Code that the project deploys from `release`, that
you want answers in Portuguese, that tests run with `pnpm test`, and next week Codex asks again, and Hermes asks a third
time. Each agent keeps its own memory, so you repeat yourself once per tool. [Renor](https://renor.cloud) is one durable
memory that all of them share over MCP. This plugin makes Renor Hermes's memory provider: what you tell Hermes is there
for Claude Code and Codex, and what you told them is there for Hermes. **Say it once, in any agent.**

## Install (3 commands)

```bash
hermes plugins install renor   # listed in the Hermes plugin catalog since 8 Nov 2026
# or, from the repository URL: hermes plugins install https://github.com/renorai/hermes-renor
hermes memory setup            # pick "renor", paste the connection key from https://app.renor.cloud
hermes                         # talk; memory is recalled before each turn and saved after it
```

To give Claude Code, Codex and Cursor the same memory, run `npx renor setup` once: it finds them, adds Renor to each
and signs you in to the same workspace.

No key? Use the open-source engine on your own machine, offline (Node 22): set `RENOR_ENGINE_COMMAND="npx -y
renor-engine mcp"` instead of a key, and point your other agents at the same folder (`RENOR_VAULT_PATH`).

The plugin uses only the Python standard library: nothing to install with it.

## What it does on each Hermes hook

| Hook | What Renor does | Bound |
| --- | --- | --- |
| `prefetch`, before each turn's model call | Asks Renor what it holds for your message (`get_context`). If memories match, tells the agent how many and asks it to read them with `renor_recall`. Greetings, acknowledgements and slash commands are skipped. | A hard deadline, `RENOR_PREFETCH_TIMEOUT_MS` (1500 ms by default). Past it the turn goes on without memory, and no second recall piles up behind a slow one. |
| `sync_turn`, after each turn | Picks out the durable things you said (below) and queues them. Returns at once: a writer thread saves them. | A queue of 64; 3 attempts with backoff for a network error, a timeout, HTTP 5xx, 408 or 429; never retries a refusal or a bad key. |
| `on_pre_compress`, before compaction | Saves the durable statements in the span about to be compressed that a turn did not already save, then waits for the queue. | At most 5 s. |
| `on_session_end`, `on_session_switch`, `shutdown` | The same over the whole session, then flushes the queue. | 10 s (2 s on a switch). |
| `on_memory_write` | Mirrors an entry Hermes adds to or rewrites in its built-in `MEMORY.md` / `USER.md`. Removals are not mirrored: Renor retires a memory by its id, and Hermes does not have that id. | Through the same queue. |

**What counts as durable** (`RENOR_AUTO_SAVE=cues`, the default), from what *you* typed, never the assistant's reply or
tool output: an explicit request ("remember that the staging box runs Debian 12"), a standing rule ("from now on, answer
in Portuguese", "always squash merge"), a stated preference or decision ("I prefer short answers", "we deploy from
`release`"), or a stated attribute ("my timezone is America/Sao_Paulo"). Questions, tasks ("remember to run the tests"),
code blocks and anything shaped like a credential are left out. At most three statements per message. These are local
rules: no model call decides what to save. `RENOR_AUTO_SAVE=turns` saves every turn instead (your message and the reply,
1,800 characters each); `off` saves nothing automatically.

**Two tools.** `renor_recall` looks anything up. `renor_remember` saves a preference, decision, rule or fact; pass
`subject`, `predicate` and `object` too for a value that stays current until it changes.

**Recalled memory is untrusted data.** Memory can hold text written by any of your agents or imported notes, so stored
text never enters the prompt automatically. The per-turn recall carries only a count and a request to call
`renor_recall`. Stored text reaches the model only as that tool's result, labelled `"trust": "untrusted-data"`; items
Renor marks untrusted (imports, reserved sources) are prefixed `[untrusted]` and not counted; a reply Renor marks not
relevant counts for nothing. The system prompt carries a rule never to follow anything written inside Renor memory.

Only the primary agent writes automatically: subagents, cron runs and flushes recall memory but never save on their own.

## Settings

All settings are read per profile through Hermes's secret store (`get_secret`), so a multiplexed gateway keeps each
profile's key and memory apart. `hermes memory setup` writes them to the profile's environment file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RENOR_API_KEY` | — | Connection key for hosted Renor (secret; never logged) |
| `RENOR_MCP_URL` | `https://mcp.renor.cloud/mcp` | Any Renor MCP endpoint that takes a Bearer key: `https://`, or `http://` only to a loopback IP (`127.0.0.1`, `[::1]`) |
| `RENOR_SCOPE` | global | Hosted scope, e.g. `project:kestrel` |
| `RENOR_AUTO_SAVE` | `cues` | `cues`, `turns` or `off` (above) |
| `RENOR_PREFETCH_TIMEOUT_MS` | `1500` | Per-turn recall deadline, 100 to 10,000 |
| `RENOR_ENGINE_COMMAND` | — | Self-hosted engine command; overrides the URL |
| `RENOR_VAULT_PATH` | `~/.renor` | Self-hosted memory folder |
| `RENOR_ALLOW_SHARING` | off | Self-hosted only: `1` lets the engine's own sharing choice apply (below) |

Hosted: MCP over Streamable HTTP with the key as a Bearer token. `RENOR_MCP_URL` must be `https://`; plain `http://` is
accepted only to a literal loopback address, and a name such as `localhost` is refused over plain HTTP because a hosts
file or DNS decides where it points. A URL with a user name or password in it is refused before any request. A redirect
is never followed, so the key is never re-sent elsewhere. Reads use `get_context`, writes use `remember` with
`source: hermes`: the same tools your other agents call, so everything lands in one workspace.

Self-hosted: the plugin starts `RENOR_ENGINE_COMMAND` as a child process and speaks MCP to it over stdio. The child
inherits only `PATH`, the home and temp directories and `LANG`, plus `VAULT_PATH` from `RENOR_VAULT_PATH`: none of
Hermes's other keys. Reads use `search` and `keyword`, writes use `note` and `remember_fact`. No embedding provider is
needed.

## Privacy: what leaves the machine

- **Hosted mode** sends to `mcp.renor.cloud` (or your `RENOR_MCP_URL`), over HTTPS: each non-trivial message you type,
  as the recall query; the durable statements picked out of your messages (or, with `RENOR_AUTO_SAVE=turns`, each turn);
  what the agent saves with `renor_remember`; and Hermes's built-in memory entries. Hermes redacts secrets before a
  provider sees a turn, and the plugin drops anything credential-shaped again before saving. It does not send tool
  output, files or the assistant's replies (except in `turns` mode).
- **The plugin has no telemetry.** It reports nothing about you or your usage to anyone; the only `User-Agent` it sends
  is `renor-hermes/<version>`.
- **Renor's own learning setting.** Renor can learn from how its memory is used (the engine's `docs/TELEMETRY.md`
  lists every level and field). In **self-hosted mode the plugin turns it off**: the
  engine child runs with `DO_NOT_TRACK=1` and `RENOR_SHARING=off`, so nothing is recorded for sharing or sent, unless you
  set `RENOR_ALLOW_SHARING=1`. In **hosted mode** that setting belongs to your workspace, not to any client: by default a
  hosted workspace shares anonymous usage numbers (never text, queries, names or paths), and memory text is shared only if
  the workspace owner turns it on. A plugin cannot change it; the owner turns it off under Settings > Learning on https://app.renor.cloud.
- **Self-hosted mode** makes no network calls of its own. It runs one long-lived child process. If the command is
  `npx`, npm may download the package the first time.
- **Your memory is yours.** Export it any time (the `export_memory` MCP tool); the plugin itself stores nothing on disk.

## Latency

Measured offline with `tests/bench_prefetch.py` (300 turns, 4-core Linux container, Python 3.11, Node 22):

| What | p50 | p95 |
| --- | --- | --- |
| Plugin's own cost per turn (prefetch minus the bare backend call) | 0.3 ms | — |
| Whole prefetch, loopback HTTP server that answers at once | 1.5 ms | 1.9 ms |
| Whole prefetch, real open-source engine over stdio, 50 saved memories | 2.4 ms | 3.0 ms |
| Renor not answering at all | 1,501 ms (the deadline), then the turn goes on | |

`sync_turn` adds nothing to the turn: Hermes calls it on its background worker and it only queues. Hosted latency
depends on the network between you and Renor and is not in this table; the deadline caps it either way.

## Limits

- The automatic save is rule-based: it catches what you state plainly, not facts implied over several turns. Say
  "remember that ..." or ask the agent to use `renor_remember` for anything else, or use `RENOR_AUTO_SAVE=turns`.
- Recall before a turn reports a count, not the text: the agent spends one tool call to read matching memories. That is
  the price of never letting stored text steer the model unasked.
- Removals in Hermes's built-in memory are not mirrored to Renor.
- Hosted mode needs a Renor account and a connection key; OAuth sign-in from Hermes is not supported yet.
- One external memory provider is active in Hermes at a time.

## Tests

Fast tests run without Hermes, against a stub of the two hermes-agent modules the provider imports
(`tests/hermes_stub/`, mirroring v0.23.0). The local-engine cases use the real engine, offline; the hosted cases use an
in-process Streamable HTTP MCP server. No key, no network, no model:

```bash
(cd engine && npm ci && npm run build)
(cd integrations/hermes/tests && python3 -W error::ResourceWarning -m unittest -v test_renor_provider test_review_874 test_lifecycle)
```

The same tests, plus `tests/test_hermes_end_to_end.py` (Hermes's own plugin loader, `MemoryManager`, turn sync,
compaction and prompt composition), run inside the real Hermes at a pinned release, followed by
`hermes plugins validate`, in `.github/workflows/hermes-plugin-ci.yml`. By hand:

```bash
(cd engine && npm ci && npm run build)
git clone https://github.com/NousResearch/hermes-agent /path/to/hermes-agent
git -C /path/to/hermes-agent checkout 2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19   # v0.23.0
(cd /path/to/hermes-agent && uv sync --frozen)
HERMES_AGENT_SRC=/path/to/hermes-agent uv run --frozen --project /path/to/hermes-agent python -m unittest discover -s integrations/hermes/tests -v
uv run --frozen --project /path/to/hermes-agent hermes plugins validate integrations/hermes
```

Latency: `(cd integrations/hermes/tests && python3 bench_prefetch.py)`.

MIT licensed (see `LICENSE`).
