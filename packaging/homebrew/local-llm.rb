# Homebrew formula for Local LLM. HEAD only: the project publishes no tagged
# releases, so Homebrew builds from the main branch.
#
#   brew install --HEAD ./packaging/homebrew/local-llm.rb
#   local-llm init      # .env sized to this Mac, plus the venv and dependencies
#   local-llm           # start the server (any deploy.py flag works)
#
# The app keeps its data (data/, logs/, adapters/, workspace/, .venv, .env)
# beside its code, and Homebrew replaces a keg on every upgrade. The wrapper
# therefore copies the code from the keg into a user-owned directory
# (LOCAL_LLM_HOME, default ~/.local/share/local-llm) whenever the keg changes
# and runs deploy.py there. packaging/README.md explains the trade-off;
# install.sh is the recommended install path.
class LocalLlm < Formula
  desc "Self-hosted multi-user LLM server, chat UI and LoRA loop for Apple Silicon"
  homepage "https://github.com/zeno-b/native-silicon-local-llm"
  head "https://github.com/zeno-b/native-silicon-local-llm.git", branch: "main"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  def install
    libexec.install Dir["*"]
    libexec.install ".env.example" if File.exist?(".env.example")
    (bin/"local-llm").write wrapper_script
    (bin/"local-llm").chmod 0755
  end

  def wrapper_script
    python = Formula["python@3.12"].opt_bin/"python3.12"
    <<~SH
      #!/bin/bash
      # local-llm: run deploy.py from a user-owned copy of the code (brew info local-llm).
      set -euo pipefail
      KEG="#{opt_libexec}"
      PY="#{python}"
      APP_HOME="${LOCAL_LLM_HOME:-$HOME/.local/share/local-llm}"
      if [ -d "$APP_HOME/.git" ]; then
        echo "local-llm: $APP_HOME is a git checkout (install.sh); run it there instead:" >&2
        echo "  cd $APP_HOME && .venv/bin/python deploy.py" >&2
        exit 2
      fi
      mkdir -p "$APP_HOME"
      CURRENT="$(cd "$KEG" && pwd -P)"
      if [ ! -f "$APP_HOME/deploy.py" ] || [ "$(cat "$APP_HOME/.brew-keg" 2>/dev/null || true)" != "$CURRENT" ]; then
        for dir in local_llm companion packaging; do
          if [ -d "$KEG/$dir" ]; then
            /usr/bin/rsync -a --delete "$KEG/$dir/" "$APP_HOME/$dir/"
          fi
        done
        for file in deploy.py bundle.py install.sh README.md .env.example; do
          if [ -f "$KEG/$file" ]; then
            cp -f "$KEG/$file" "$APP_HOME/$file"
          fi
        done
        echo "$CURRENT" > "$APP_HOME/.brew-keg"
      fi
      cd "$APP_HOME"
      exec "$PY" deploy.py "$@"
    SH
  end

  def caveats
    <<~EOS
      local-llm runs from a copy of the code in ~/.local/share/local-llm
      (set LOCAL_LLM_HOME to move it). The venv, .env, database, logs and
      adapters live there too, so they survive upgrades and `brew uninstall`.

      First run:
        local-llm init          # add --multi-user to require login
        local-llm               # or: local-llm service install

      After `brew upgrade --fetch-HEAD local-llm`, the next local-llm command
      refreshes the copy; restart a running service with
        local-llm service install

      install.sh is the recommended install path (a plain git checkout that
      `git pull` updates); see packaging/README.md in the repository.
    EOS
  end

  test do
    ENV["LOCAL_LLM_HOME"] = (testpath/"app").to_s
    output = shell_output("#{bin}/local-llm init --dry-run --non-interactive")
    assert_match "MODEL_ID=", output
    assert_path_exists testpath/"app/deploy.py"
  end
end
