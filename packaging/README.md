# Packaging

Two ways to install Local LLM on an Apple Silicon Mac. **install.sh is the
recommended one**; the Homebrew formula exists for people who manage every tool
through `brew`.

## install.sh (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/zeno-b/native-silicon-local-llm/main/install.sh | bash
# options go after `bash -s --`:
curl -fsSL https://raw.githubusercontent.com/zeno-b/native-silicon-local-llm/main/install.sh | bash -s -- --multi-user --service
```

It checks for macOS on native arm64, finds Python 3.10+ (suggesting
`brew install python@3.12` when there is none; it never uses sudo), clones the
repository into `~/local-llm` (`--dir` to change it) or fast-forwards an existing
checkout, then runs `deploy.py init`. init creates `./.venv`, installs the
dependencies through the app's own bootstrap (the same code path as the first
server launch) and writes `.env` for this Mac's RAM. `--service` adds the launchd
LaunchAgent; `--join URL --code CODE` turns the Mac into a secondary node of an
existing primary. `--dry-run` prints every action without running it.

Re-running is safe: an existing `.env` is kept and only the dependencies are
refreshed. Passing init flags (or `--force`) regenerates the configuration, and
without `--force` init writes `.env.new` beside the existing file and prints what
differs.

The result is a plain git checkout: `git pull` updates it, and the venv, `.env`,
database, logs and adapters stay where they are.

## Homebrew (HEAD only)

```bash
brew install --HEAD ./packaging/homebrew/local-llm.rb    # from a checkout
local-llm init
local-llm                                                # any deploy.py flag works
```

The formula is HEAD only because the project publishes no tagged releases. It
depends on `python@3.12`, requires macOS on arm64 and installs a `local-llm`
wrapper.

### Where the data lives, and the trade-off

The app keeps everything it writes **beside its code**: `core.ROOT` is the
directory that holds `deploy.py`, and `data/`, `logs/`, `adapters/`,
`workspace/`, `.venv` and `.env` all hang off it. Homebrew, by contrast, treats a
keg as disposable: `brew upgrade` installs the new version into a new Cellar
directory and deletes the old one. Running `deploy.py` straight from the keg would
therefore lose the database, the trained adapters and the venv on every upgrade.

The wrapper avoids that by copying the code (`deploy.py`, `local_llm/`, and a few
companions) from the keg into a user-owned directory, `LOCAL_LLM_HOME`
(default `~/.local/share/local-llm`), whenever the keg changes, and running it
there. That directory is effectively a checkout without git: the data survives
upgrades and `brew uninstall`, at these costs:

- **Two copies of the code.** The keg holds one, `LOCAL_LLM_HOME` the one that
  actually runs. A local edit in `LOCAL_LLM_HOME` is overwritten by the next
  upgrade.
- **Upgrades apply on the next `local-llm` command.** A launchd service started
  with `local-llm service install` runs the copy directly, so after
  `brew upgrade --fetch-HEAD local-llm` run any `local-llm` command (or
  `local-llm service install` again) to refresh the copy and restart it.
- **`brew uninstall` leaves the data behind** on purpose. Delete
  `LOCAL_LLM_HOME` yourself to remove it.
- **No `git pull`.** Updates come only through `brew upgrade --fetch-HEAD`.

The wrapper refuses to run when `LOCAL_LLM_HOME` is a git checkout (one made by
install.sh), so the two install paths never write over each other.

## Multi-node bootstrap

```bash
# primary (e.g. the Mac mini)
curl -fsSL .../install.sh | bash -s -- --cluster primary --service
# then in its UI: Settings > Cluster > Pair a node, which shows a one-time code

# each secondary (e.g. the Mac Studio)
curl -fsSL .../install.sh | bash -s -- --join http://mac-mini.local:8000 --code CODE --service
```

`--cluster` turns login on, binds all interfaces and, on the primary, generates
`NODE_TOKEN`. `--join` runs the existing `deploy.py cluster join` flow, which
writes the node token the primary issues into the secondary's `.env`.

## Files

| file | what |
| --- | --- |
| `../install.sh` | the curl installer |
| `homebrew/local-llm.rb` | the HEAD-only formula and its `local-llm` wrapper |
| `../local_llm/setup_cmds.py` | `deploy.py init`, `deploy.py service`, the `--doctor` install checks |
| `../local_llm/envfile.py` | the `.env` loader deploy.py runs before anything else |
