#!/usr/bin/env bash
# Local LLM installer for Apple Silicon Macs.
#
#   curl -fsSL https://raw.githubusercontent.com/zeno-b/native-silicon-local-llm/main/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --multi-user --service
#   ./install.sh --join http://mac-mini.local:8000 --code 123-456     # add a secondary Mac
#
# What it does, in order; every step is safe to re-run:
#   1. checks for macOS on Apple Silicon (native arm64, not Rosetta);
#   2. finds Python 3.10+ built for arm64 (suggests `brew install python@3.12`);
#   3. clones the repository into --dir (default ~/local-llm) or fast-forwards
#      an existing checkout; run from inside a checkout, it uses that one;
#   4. runs `deploy.py init`, which creates ./.venv and installs dependencies
#      through the app's own bootstrap, then writes .env sized to this Mac's
#      RAM. An existing .env is kept: a re-run only refreshes dependencies,
#      and an explicit init flag writes .env.new beside it unless --force;
#   5. with --service, installs the launchd LaunchAgent (starts at login).
#
# It never uses sudo. --dry-run prints every action instead of running it.
set -euo pipefail

REPO_DEFAULT="https://github.com/zeno-b/native-silicon-local-llm.git"

usage() {
  cat <<'EOF'
Usage: install.sh [options] [-- extra deploy.py init flags]

Where and what:
  --dir DIR             install into DIR (default: ~/local-llm)
  --repo URL            clone from URL (default: the upstream GitHub repo)
  --branch NAME         clone this branch
  --python PATH         use this Python 3.10+ (arm64) interpreter
  --dry-run             print every action without running it

Configuration (passed to `deploy.py init`):
  --non-interactive     ask nothing (implied when there is no terminal)
  --multi-user          require login and generate an admin password
  --cluster ROLE        primary | secondary
  --primary-url URL     with --cluster secondary
  --model ID            use this model instead of the RAM-based recommendation
  --host HOST           WEB_HOST to write
  --admin-user NAME     admin user name for --multi-user
  --force               regenerate .env (the old one is kept as .env.bak)

Extras:
  --join URL --code C   bootstrap a secondary node: pair with the primary at URL
  --service             install the launchd service (starts now and at login)
  -h, --help            this help
EOF
}

say()  { printf '%s\n' "$*"; }
note() { printf 'note: %s\n' "$*" >&2; }
die()  { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

# Print a command the way a shell would read it back.
show() { printf '+'; printf ' %q' "$@"; printf '\n'; }

run() {
  if [ "$DRY" = 1 ]; then show "$@"; return 0; fi
  "$@"
}

# Run a command inside the install directory; stdin is the terminal when
# there is one (under `curl | bash` stdin is the script itself).
run_in_dir() {
  if [ "$DRY" = 1 ]; then printf '+ (cd %q &&' "$DIR"; printf ' %q' "$@"; printf ')\n'; return 0; fi
  if [ "$TTY" = 1 ]; then
    (cd "$DIR" && "$@" </dev/tty)
  else
    (cd "$DIR" && "$@" </dev/null)
  fi
}

clt_ok() { xcode-select -p >/dev/null 2>&1; }

check_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  [ "$os" = "Darwin" ] || die "this installer is for macOS on Apple Silicon (found $os)."
  if [ "$arch" != "arm64" ]; then
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
      die "this shell runs under Rosetta (x86_64). Open a native arm64 terminal and run the installer again."
    fi
    die "Apple Silicon (arm64) is required; this Mac reports $arch."
  fi
}

# "MAJOR.MINOR MACHINE" for an interpreter, or failure.
python_info() {
  "$1" -c 'import sys, platform; print("%d.%d %s" % (sys.version_info[0], sys.version_info[1], platform.machine()))' 2>/dev/null
}

find_python() {
  local cand path info ver mach major minor
  if [ -n "$PYTHON" ] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    die "--python $PYTHON: no such interpreter."
  fi
  for cand in ${PYTHON:+"$PYTHON"} python3 python3.13 python3.12 python3.11 python3.10 \
              /opt/homebrew/bin/python3; do
    path="$(command -v "$cand" 2>/dev/null)" || continue
    # /usr/bin/python3 is a stub that opens an install dialog without the
    # Command Line Tools; skip it rather than pop a window mid-install.
    if [ "$path" = "/usr/bin/python3" ] && ! clt_ok; then continue; fi
    info="$(python_info "$path")" || continue
    ver="${info%% *}"
    mach="${info#* }"
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$mach" = "arm64" ]; then
      PY="$path"
      PY_VERSION="$ver"
      return 0
    fi
    note "skipping $path: Python $ver on $mach (need 3.10+ on arm64)"
    if [ -n "$PYTHON" ] && [ "$cand" = "$PYTHON" ]; then
      die "--python $PYTHON is Python $ver on $mach; need 3.10 or newer on arm64."
    fi
  done
  return 1
}

check_git() {
  local path
  path="$(command -v git 2>/dev/null)" || die "git is missing. Install the Command Line Tools (xcode-select --install) or Homebrew git (brew install git), then re-run."
  if [ "$path" = "/usr/bin/git" ] && ! clt_ok; then
    die "git needs the Command Line Tools: run xcode-select --install, then re-run."
  fi
}

sync_repo() {
  if [ "$IN_PLACE" = 1 ]; then
    say "Using this checkout as is: $DIR"
    return 0
  fi
  if [ -d "$DIR/.git" ]; then
    local origin
    origin="$(git -C "$DIR" remote get-url origin 2>/dev/null || true)"
    if [ -n "$origin" ] && [ "$origin" != "$REPO" ]; then
      note "$DIR tracks $origin, not $REPO; updating it from its own origin."
    fi
    if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]; then
      note "$DIR has local changes, so it was not updated (commit or stash them, then re-run)."
      return 0
    fi
    say "Updating $DIR"
    run git -C "$DIR" pull --ff-only --quiet || note "update failed; continuing with the current checkout."
  elif [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
    die "$DIR exists and is not a git checkout; pick another --dir."
  else
    say "Cloning $REPO into $DIR"
    run git clone --quiet ${BRANCH:+--branch "$BRANCH"} "$REPO" "$DIR"
  fi
}

main() {
  REPO="${LOCAL_LLM_REPO:-$REPO_DEFAULT}"
  DIR="${LOCAL_LLM_DIR:-$HOME/local-llm}"
  DIR_SET=0
  [ -n "${LOCAL_LLM_DIR:-}" ] && DIR_SET=1
  BRANCH=""
  PYTHON=""
  DRY=0
  SERVICE=0
  FORCE=0
  JOIN_URL=""
  JOIN_CODE=""
  EXPLICIT_INIT=0
  IN_PLACE=0
  INIT_ARGS=()

  while [ $# -gt 0 ]; do
    case "$1" in
      --dir|--repo|--branch|--python|--cluster|--primary-url|--model|--host|--admin-user|--join|--code)
        [ $# -ge 2 ] || die "$1 needs a value (see --help)."
        case "$1" in
          --dir) DIR="$2"; DIR_SET=1 ;;
          --repo) REPO="$2" ;;
          --branch) BRANCH="$2" ;;
          --python) PYTHON="$2" ;;
          --join) JOIN_URL="$2" ;;
          --code) JOIN_CODE="$2" ;;
          *) INIT_ARGS+=("$1" "$2"); EXPLICIT_INIT=1 ;;
        esac
        shift 2 ;;
      --multi-user) INIT_ARGS+=("$1"); EXPLICIT_INIT=1; shift ;;
      --non-interactive) INIT_ARGS+=("$1"); shift ;;
      --force) INIT_ARGS+=("$1"); FORCE=1; shift ;;
      --service) SERVICE=1; shift ;;
      --dry-run) DRY=1; shift ;;
      -h|--help) usage; exit 0 ;;
      --) shift; if [ $# -gt 0 ]; then INIT_ARGS+=("$@"); EXPLICIT_INIT=1; fi; break ;;
      *) die "unknown option: $1 (see --help)." ;;
    esac
  done

  if [ -n "$JOIN_URL" ]; then
    [ -n "$JOIN_CODE" ] || die "--join needs --code (the primary shows one under Settings > Cluster)."
    INIT_ARGS+=(--cluster secondary --primary-url "$JOIN_URL" --code "$JOIN_CODE")
    EXPLICIT_INIT=1
  elif [ -n "$JOIN_CODE" ]; then
    INIT_ARGS+=(--code "$JOIN_CODE")
  fi

  # Run from inside a checkout (./install.sh), use that checkout.
  local src="${BASH_SOURCE[0]:-}" here=""
  if [ "$DIR_SET" = 0 ] && [ -n "$src" ] && [ -f "$src" ]; then
    here="$(cd "$(dirname "$src")" && pwd -P)"
    if [ -f "$here/deploy.py" ] && [ -d "$here/local_llm" ]; then
      DIR="$here"
      IN_PLACE=1
    fi
  fi

  TTY=0
  if [ "$DRY" = 0 ] && (: </dev/tty) 2>/dev/null; then TTY=1; fi
  if [ "$TTY" = 0 ]; then
    case " ${INIT_ARGS[*]-} " in
      *" --non-interactive "*) ;;
      *) INIT_ARGS+=(--non-interactive) ;;
    esac
  fi

  [ "$DRY" = 1 ] && say "Dry run: nothing below is executed."
  check_platform
  PY=""
  PY_VERSION=""
  find_python || die "no Python 3.10+ for arm64 found. Install one with Homebrew (https://brew.sh): brew install python@3.12, then re-run this installer."
  say "Python $PY_VERSION: $PY"
  [ "$IN_PLACE" = 1 ] || check_git
  sync_repo

  if [ -f "$DIR/.env" ] && [ "$FORCE" = 0 ] && [ "$EXPLICIT_INIT" = 0 ]; then
    say "Keeping $DIR/.env; refreshing dependencies only (pass --force or init flags to regenerate it)."
    run_in_dir "$PY" deploy.py init --deps-only \
      || die "dependency install failed (see above). Fix it, then re-run this installer."
  else
    say "Configuring for this Mac (creates .venv and installs dependencies on the first run)."
    run_in_dir "$PY" deploy.py init ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
      || die "deploy.py init failed (see above). Fix it, then re-run this installer."
  fi

  if [ "$SERVICE" = 1 ]; then
    run_in_dir "$DIR/.venv/bin/python" deploy.py service install \
      || die "the launchd service did not install (see above)."
  fi

  say ""
  if [ "$DRY" = 1 ]; then
    say "Dry run complete: Local LLM would be installed in $DIR."
  else
    say "Local LLM is installed in $DIR."
  fi
  if [ "$SERVICE" = 1 ]; then
    say "  It runs as a launchd service:  cd $DIR && .venv/bin/python deploy.py service status"
  else
    say "  Start it:      cd $DIR && .venv/bin/python deploy.py"
    say "  At login:      cd $DIR && .venv/bin/python deploy.py service install"
  fi
  say "  Check health:  cd $DIR && .venv/bin/python deploy.py --doctor"
}

main "$@"
