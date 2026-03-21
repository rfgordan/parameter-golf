#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:?repo url required}"
GIT_REF="${2:?git ref required}"
REMOTE_ROOT="${3:?remote root required}"
REMOTE_PAYLOAD_PATH="${4:?remote payload path required}"
REMOTE_RUN_DIR="${5:?remote run dir required}"
TMUX_SESSION="${6:?tmux session required}"

DATASET_VARIANT="${PARAMETER_GOLF_DATASET_VARIANT:-sp1024}"
TRAIN_SHARDS="${PARAMETER_GOLF_TRAIN_SHARDS:-80}"
REPO_DIR="${REMOTE_ROOT}/repo"
LAUNCHER_PATH="${REMOTE_ROOT}/launch_job.sh"

mkdir -p "${REMOTE_ROOT}" "${REMOTE_RUN_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update >/dev/null
  apt-get install -y tmux >/dev/null
fi

if ! command -v git >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update >/dev/null
  apt-get install -y git >/dev/null
fi

cat > "${LAUNCHER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

REPO_URL=$(printf '%q' "${REPO_URL}")
GIT_REF=$(printf '%q' "${GIT_REF}")
REMOTE_ROOT=$(printf '%q' "${REMOTE_ROOT}")
REMOTE_PAYLOAD_PATH=$(printf '%q' "${REMOTE_PAYLOAD_PATH}")
REMOTE_RUN_DIR=$(printf '%q' "${REMOTE_RUN_DIR}")
DATASET_VARIANT=$(printf '%q' "${DATASET_VARIANT}")
TRAIN_SHARDS=$(printf '%q' "${TRAIN_SHARDS}")
REPO_DIR=\${REMOTE_ROOT}/repo

mkdir -p "\${REMOTE_RUN_DIR}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "\${REMOTE_RUN_DIR}/.runpod_started_at"
printf '%s\n' "bootstrap_started" > "\${REMOTE_RUN_DIR}/.runpod_status"

rm -rf "\${REPO_DIR}"
git clone --depth 1 "\${REPO_URL}" "\${REPO_DIR}"
cd "\${REPO_DIR}"
git fetch --depth 1 origin "\${GIT_REF}" >/dev/null 2>&1 || true
git checkout "\${GIT_REF}" >/dev/null 2>&1 || git checkout FETCH_HEAD >/dev/null 2>&1

printf '%s\n' "data_setup" > "\${REMOTE_RUN_DIR}/.runpod_status"
python3 data/cached_challenge_fineweb.py --variant "\${DATASET_VARIANT}" --train-shards "\${TRAIN_SHARDS}"

printf '%s\n' "training" > "\${REMOTE_RUN_DIR}/.runpod_status"
set +e
PARAMETER_GOLF_FIXED_RUN_DIR="\${REMOTE_RUN_DIR}" python3 ops/runpod/worker.py --payload-file "\${REMOTE_PAYLOAD_PATH}" > "\${REMOTE_RUN_DIR}/launcher_output.json" 2>&1
rc=\$?
set -e
printf '%s\n' "\${rc}" > "\${REMOTE_RUN_DIR}/.runpod_exit_code"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "\${REMOTE_RUN_DIR}/.runpod_finished_at"
if [ "\${rc}" -eq 0 ]; then
  printf '%s\n' "success" > "\${REMOTE_RUN_DIR}/.runpod_status"
else
  printf '%s\n' "train_failed" > "\${REMOTE_RUN_DIR}/.runpod_status"
fi
EOF

chmod +x "${LAUNCHER_PATH}"
printf '%s\n' "${TMUX_SESSION}" > "${REMOTE_RUN_DIR}/.runpod_tmux_session"
tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && tmux kill-session -t "${TMUX_SESSION}"
tmux new-session -d -s "${TMUX_SESSION}" "${LAUNCHER_PATH}"
printf 'started tmux session %s\n' "${TMUX_SESSION}"
