# MCP servers: what's worth wiring, and what isn't

MCP is a **research and execution** layer, not a latency layer. Every MCP call is
a round trip through an LLM turn — hundreds of milliseconds minimum, usually
seconds. Nothing in the 16:05 hot path should touch MCP.

The split that matters:

| Layer | Who runs it | Latency | What it's for |
|---|---|---|---|
| `signalsniper` daemon | your machine, always on | sub-ms + network | detection, gating, sizing |
| MCP servers | Claude, on demand | 0.5–5s | pre-market prep, reading filings, placing/managing orders, post-mortem |

Use MCP to **decide what to watch** before the bell and to **act on** what the
daemon flags. Do not put it between the filing and the order.

---

## Tier 1 — install these

### Alpaca (execution + market data)

Official, ~70 tools, and it does both trading and data.

```bash
claude mcp add alpaca --scope user --transport stdio uvx alpaca-mcp-server \
  --env ALPACA_API_KEY=your_key \
  --env ALPACA_SECRET_KEY=your_secret \
  --env ALPACA_PAPER_TRADE=true
```

Keep `ALPACA_PAPER_TRADE=true` until you've watched a full session of alerts and
agree with them. Flipping it is a one-word change; unflipping a bad fill isn't.

You can narrow the tool surface, which meaningfully improves how reliably the
model picks the right one:

```json
{ "env": { "ALPACA_TOOLSETS": "stock-data,account,trading" } }
```

### Massive (formerly Polygon.io) — market data

**Note the rename**: Polygon.io became Massive on 2025-10-30. `api.polygon.io`
still works and SDKs stay backward compatible, but the MCP server ships as
`mcp_massive`. Install as a tool rather than `uvx` — dependency downloads on
every start cause timeouts.

```bash
uv tool install "mcp_massive @ git+https://github.com/massive-com/mcp_massive@v0.10.0"
```

```json
{
  "mcpServers": {
    "massive": {
      "command": "<path printed by uv tool install>",
      "env": { "MASSIVE_API_KEY": "<key>", "HOME": "<your home dir>" }
    }
  }
}
```

Three composable tools (`search_endpoints`, `call_api`, `query_data`) rather than
a wide surface — `query_data` runs SQL over pulled results, which is genuinely
useful for pre-market screening ("every name in the watchlist whose 20d ADV is
under 500k shares" is one query, not fifteen calls).

### Fetch (reading the actual filing)

The single highest-value MCP call in this whole workflow: pull Exhibit 99.1 off
EDGAR and read what the press release actually says while the crowd is still on
the headline.

```bash
claude mcp add fetch --scope user --transport stdio uvx mcp-server-fetch
```

---

## Tier 2 — worth it if you'll use them

- **Filesystem** — point at this repo so Claude can read the alert log and the
  linkage graph directly during a session.
  ```bash
  claude mcp add filesystem --scope user --transport stdio \
    npx -y @modelcontextprotocol/server-filesystem /path/to/SignallSniper
  ```
- **Sequential-thinking** — helps on the multi-hop reasoning ("AWS accelerated,
  so which of ANET/VRT/MRVL has the tightest revenue attribution?").
- **Memory / knowledge-graph** — persists calibration notes across sessions. Only
  useful if you actually write post-mortems into it.

---

## Skip these

- **Any "AI trading signal" MCP server.** You're building the signal. A wrapper
  around someone else's sentiment score adds latency and an opaque dependency.
- **Reddit/X sentiment MCP.** Social sentiment lags price on liquid names. The
  original `main.py` in this repo did exactly this — see `docs/DATA_SOURCES.md`
  for why it was demoted.
- **Anything that wants your broker password** rather than a scoped API key.

---

## Config sanity

- Scope keys to what they need. An Alpaca key with trading enabled sitting in a
  shell profile is a bad trade you haven't made yet.
- Paper and live keys are different credentials on Alpaca. Keep two profiles;
  don't edit one string back and forth.
- After adding servers, `claude mcp list` to confirm they actually started. A
  server that silently failed to launch looks identical to one you forgot about.
