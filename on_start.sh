bash -lc '
set -euo pipefail

# ── 0 · Env toggles ──────────────────────────────────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_ROOT_USER_ACTION=ignore
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# ── 1 · System packages ─────────────────────────────────────────────────────
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git nano ninja-build build-essential python3-dev python3-venv \
  libnuma1 libnuma-dev pkg-config curl ca-certificates

# ── 2 · Git identity (optional) ─────────────────────────────────────────────
[ -n "${GIT_USER_NAME:-}"  ] && git config --global user.name  "$GIT_USER_NAME"
[ -n "${GIT_USER_EMAIL:-}" ] && git config --global user.email "$GIT_USER_EMAIL"

# ── 3 · Repo ────────────────────────────────────────────────────────────────
cd /workspace
if [ -d re-arc/.git ]; then
  git -C re-arc pull --ff-only
else
  if [ -n "${GITHUB_PAT:-}" ]; then
    git clone "https://${GITHUB_PAT}@github.com/TrelisResearch/re-arc.git"
  else
    git clone https://github.com/TrelisResearch/re-arc.git
  fi
fi
cd re-arc
uv sync

bash init.sh
'