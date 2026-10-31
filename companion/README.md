# Menu-bar companion

Two ways to see the Local LLM server's health from the macOS menu bar without
keeping a browser tab open. Both read one route, `GET /api/status/summary`, and
pull the existing admin levers (`POST /api/model/restart`,
`POST /api/tasks/{id}/cancel`, `POST /api/retrain`).

| | LocalLLM Bar (native) | SwiftBar / xbar plugin |
|---|---|---|
| Source | `macos/LocalLLMBar.swift` | `swiftbar/local-llm.30s.py` |
| Needs | Xcode command line tools (swiftc), macOS 13+ | SwiftBar or xbar, python3 |
| Refresh | every 5 s, backing off to 60 s while unreachable | every 30 s |
| Token | login Keychain | `$LLM_API_TOKEN` or the CLI's credentials file |

## The token

Create a personal API token in the web UI (Settings, API tokens) with the
**status** scope. Add **control** (administrators only) to enable Restart Model
and Cancel Running Tasks; Retrain Now appears only for an **admin**-scoped token,
because `/api/retrain` is not one of the control scope's routes. A chat-only
token is refused with 403. With sign-in turned off on the server, no token is
needed.

What the summary shows depends on who asks, exactly like `/api/health`: every
caller sees the model state, backend, queue, its own pending approvals and its
own last-hour figures; administrators also see cluster nodes, running tasks,
retraining and everyone's last-hour figures. The `controls` block tells the
client which levers this credential may pull, so menu items are greyed out
instead of failing with 403. No secrets, node URLs or paths are in it.

## Native app

```sh
python3 deploy.py companion build             # dist/LocalLLM Bar.app (this Mac's arch)
python3 deploy.py companion build --arch universal
python3 deploy.py companion install           # copy to ~/Applications
python3 deploy.py companion install --login   # ...and start it at login (LaunchAgent)
python3 deploy.py companion uninstall         # remove the app and the login item
open ~/Applications/"LocalLLM Bar.app"
```

`build` compiles the single Swift file with swiftc (no Xcode project, no
third-party code), renders `macos/Info.plist` (`LSUIElement` so it lives in the
menu bar only, bundle id `local.llm.bar`) and signs the bundle ad hoc so the
Keychain recognises the app across rebuilds. `--dry-run` prints the commands,
`--no-sign` skips codesign, `--no-optimize` compiles faster for a quick check.

Open **Settings...** in its menu to set the server URL (default
`http://127.0.0.1:8000`) and paste the token; the token is stored in the login
Keychain under the service `local.llm.bar`. On first run both default from the
CLI's saved sign-in at `~/.config/local-llm/credentials.json` if it exists.

The menu: a coloured dot with a short label (ready, loading, queue n, error,
offline, sign in), status lines (model, adapter, backend and capabilities,
queue, last-hour requests with TTFT p50 and decode tok/s, nodes, tasks,
retraining), then Open Web UI, Approvals Pending (opens the UI), Restart
Model (asks first), Cancel Running Tasks (asks first), Copy API URL (the
OpenAI-compatible `/v1` base), Refresh Now, Settings, Quit.

## SwiftBar / xbar plugin

```sh
python3 deploy.py companion plugin                 # into SwiftBar's (or xbar's) plugin folder
python3 deploy.py companion plugin --dir ~/plugins
```

It reads `LLM_URL` and `LLM_API_TOKEN`, else the CLI's credentials file. The
token never appears in the plugin's output (SwiftBar caches and logs it): the
action lines call the plugin itself, which runs curl with the Authorization
header passed on curl's stdin, so the token is not on any command line either.
