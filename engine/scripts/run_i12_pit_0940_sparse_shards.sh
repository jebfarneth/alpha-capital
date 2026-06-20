#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-i12pit_0940_sparse}"
SCHEMA="${SCHEMA:-scratch_i12_pit_m1_0940_sparse_20260618}"
MINUTE_PATH_MODE="sparse_zero_fill"
DECISION_TIME="09:40"
SOURCE_HUR_SCHEMA="${SOURCE_HUR_SCHEMA:-public}"
MAX_NO_PROGRESS_MINUTES="${MAX_NO_PROGRESS_MINUTES:-20}"
MAX_RESUMES="${MAX_RESUMES:-5}"
REPLACE_STALE="${REPLACE_STALE:-0}"
REPLACE_RUNNING="${REPLACE_RUNNING:-0}"
ONLY_SHARD="${ONLY_SHARD:-}"
ONLY_WINDOW="${ONLY_WINDOW:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

SHARDS=(
  "may01_07:2026-05-01:2026-05-07:artifacts/stage0/i12_pit_0940_sparse_may01_07.json"
  "may08_14:2026-05-08:2026-05-14:artifacts/stage0/i12_pit_0940_sparse_may08_14.json"
  "may15_21:2026-05-15:2026-05-21:artifacts/stage0/i12_pit_0940_sparse_may15_21.json"
  "may22_29:2026-05-22:2026-05-29:artifacts/stage0/i12_pit_0940_sparse_may22_29.json"
  "jun01_05:2026-06-01:2026-06-05:artifacts/stage0/i12_pit_0940_sparse_jun01_05.json"
)

matched_shards=0
launched_windows=0
skipped_running_windows=0
replaced_stale_windows=0
replaced_running_windows=0

quote_cmd() {
  printf "%q " "$@"
}

regex_escape() {
  printf "%s" "$1" | sed 's/[][(){}.^$*+?|\\]/\\&/g'
}

normalize_tmux_pane_start_command() {
  printf "%s" "$1" | sed -E 's/\\+[[:space:]]+/ /g'
}

command_has_arg_value() {
  local command="$1"
  local flag
  local value
  local pattern
  local normalized_command
  normalized_command="$(normalize_tmux_pane_start_command "${command}")"
  flag="$(regex_escape "$2")"
  value="$(regex_escape "$3")"
  pattern="(^|[[:space:]])${flag}([[:space:]]+|=)${value}($|[[:space:]])"
  [[ "${normalized_command}" =~ ${pattern} ]]
}

command_has_arg() {
  local command="$1"
  local flag
  local pattern
  local normalized_command
  normalized_command="$(normalize_tmux_pane_start_command "${command}")"
  flag="$(regex_escape "$2")"
  pattern="(^|[[:space:]])${flag}($|[[:space:]])"
  [[ "${normalized_command}" =~ ${pattern} ]]
}

selector_enabled() {
  [[ -n "${ONLY_SHARD}" || -n "${ONLY_WINDOW}" ]]
}

print_valid_selectors() {
  echo "Valid sparse shard selectors:" >&2
  for shard in "${SHARDS[@]}"; do
    IFS=":" read -r name _start_date _end_date _progress_artifact <<<"${shard}"
    echo "  ONLY_SHARD=${name} or ONLY_WINDOW=sparse_${name}" >&2
  done
}

validate_replacement_scope() {
  local selector_count=0
  [[ -n "${ONLY_SHARD}" ]] && selector_count=$((selector_count + 1))
  [[ -n "${ONLY_WINDOW}" ]] && selector_count=$((selector_count + 1))
  if [[ "${REPLACE_RUNNING}" == "1" && "${selector_count}" -ne 1 ]]; then
    echo "ERROR: REPLACE_RUNNING=1 requires exactly one of ONLY_SHARD or ONLY_WINDOW." >&2
    echo "Example: REPLACE_RUNNING=1 ONLY_SHARD=may15_21 $0" >&2
    echo "Example: REPLACE_RUNNING=1 ONLY_WINDOW=sparse_may15_21 $0" >&2
    exit 1
  fi
}

validate_selector_matches() {
  local matched=0
  local name window shard
  if ! selector_enabled; then
    return 0
  fi
  for shard in "${SHARDS[@]}"; do
    IFS=":" read -r name _start_date _end_date _progress_artifact <<<"${shard}"
    window="sparse_${name}"
    if shard_selected "${name}" "${window}"; then
      matched=$((matched + 1))
    fi
  done
  if [[ "${matched}" -eq 0 ]]; then
    echo "ERROR: selector matched zero shards: ONLY_SHARD='${ONLY_SHARD}' ONLY_WINDOW='${ONLY_WINDOW}'" >&2
    print_valid_selectors
    exit 1
  fi
}

shard_selected() {
  local shard_name="$1"
  local window="$2"
  if [[ -n "${ONLY_SHARD}" && "${shard_name}" != "${ONLY_SHARD}" ]]; then
    return 1
  fi
  if [[ -n "${ONLY_WINDOW}" && "${window}" != "${ONLY_WINDOW}" ]]; then
    return 1
  fi
  return 0
}

require_env_file() {
  if [[ ! -f "${ENGINE_DIR}/.env" ]]; then
    echo "ERROR: ${ENGINE_DIR}/.env is required" >&2
    exit 1
  fi
}

run_schema_preflight() {
  echo "preflighting scratch schema ${SCHEMA} for ${MINUTE_PATH_MODE}"
  (
    set -Eeuo pipefail
    cd "${ENGINE_DIR}"
    require_env_file
    set -a
    source .env
    set +a
    "${PYTHON_BIN}" -m alpha.jobs.run_i12_pit_rebuild \
      --schema "${SCHEMA}" \
      --create-tables \
      --source-hur-schema "${SOURCE_HUR_SCHEMA}" \
      --decision-time "${DECISION_TIME}" \
      --minute-path-mode "${MINUTE_PATH_MODE}" \
      --preflight-only
  )
}

worker_shell_command() {
  local -a run_cmd=("$@")
  local inner
  inner="set -Eeuo pipefail; cd $(printf "%q" "${ENGINE_DIR}"); "
  inner+="if [[ ! -f .env ]]; then echo 'ERROR: .env is required' >&2; exit 1; fi; "
  inner+="set -a; source .env; set +a; exec $(quote_cmd "${run_cmd[@]}")"
  printf "exec bash -lc %q" "${inner}"
}

window_running_expected() {
  local window="$1"
  local start_date="$2"
  local end_date="$3"
  local progress_artifact="$4"
  local artifact_base
  local pane_dead pane_command pane_start
  artifact_base="${progress_artifact%.json}"
  while IFS=$'\t' read -r pane_dead pane_command pane_start; do
    [[ "${pane_dead}" == "0" ]] || continue
    [[ "${pane_start}" == *"run_i12_pit_shard_supervised.sh"* ]] || continue
    command_has_arg_value "${pane_start}" "--schema" "${SCHEMA}" || continue
    command_has_arg_value "${pane_start}" "--source-hur-schema" "${SOURCE_HUR_SCHEMA}" || continue
    command_has_arg_value "${pane_start}" "--start-date" "${start_date}" || continue
    command_has_arg_value "${pane_start}" "--end-date" "${end_date}" || continue
    command_has_arg_value "${pane_start}" "--decision-time" "${DECISION_TIME}" || continue
    command_has_arg_value "${pane_start}" "--minute-path-mode" "${MINUTE_PATH_MODE}" || continue
    command_has_arg_value "${pane_start}" "--artifact-base" "${artifact_base}" || continue
    command_has_arg_value "${pane_start}" "--max-no-progress-minutes" "${MAX_NO_PROGRESS_MINUTES}" || continue
    command_has_arg_value "${pane_start}" "--max-resumes" "${MAX_RESUMES}" || continue
    return 0
  done < <(tmux list-panes -t "${SESSION}:${window}" -F "#{pane_dead}\t#{pane_current_command}\t#{pane_start_command}" 2>/dev/null || true)
  return 1
}

handle_existing_window() {
  local window="$1"
  local start_date="$2"
  local end_date="$3"
  local progress_artifact="$4"
  if ! tmux list-windows -t "${SESSION}" -F "#{window_name}" | grep -Fxq "${window}"; then
    return 0
  fi
  if window_running_expected "${window}" "${start_date}" "${end_date}" "${progress_artifact}"; then
    if [[ "${REPLACE_RUNNING}" == "1" ]]; then
      echo "WARNING: replacing running expected shard window: ${SESSION}:${window}" >&2
      echo "WARNING: verify pg_stat_activity/progress artifacts before using REPLACE_RUNNING=1." >&2
      tmux kill-window -t "${SESSION}:${window}"
      replaced_running_windows=$((replaced_running_windows + 1))
      return 0
    fi
    echo "window already running expected shard: ${SESSION}:${window}"
    echo "Set REPLACE_RUNNING=1 to kill/recreate an expected-command shard after diagnosing a hang." >&2
    skipped_running_windows=$((skipped_running_windows + 1))
    return 1
  fi
  if [[ "${REPLACE_STALE}" == "1" ]]; then
    echo "replacing stale window: ${SESSION}:${window}"
    tmux kill-window -t "${SESSION}:${window}"
    replaced_stale_windows=$((replaced_stale_windows + 1))
    return 0
  fi
  echo "ERROR: stale or unexpected tmux window exists: ${SESSION}:${window}" >&2
  echo "Set REPLACE_STALE=1 to kill/recreate it after verifying it is safe." >&2
  tmux list-panes -t "${SESSION}:${window}" -F "#{pane_index} dead=#{pane_dead} cmd=#{pane_current_command} start=#{pane_start_command}" >&2 || true
  exit 1
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required" >&2
  exit 1
fi

validate_replacement_scope
validate_selector_matches
run_schema_preflight

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n control -c "${ENGINE_DIR}"
fi

for shard in "${SHARDS[@]}"; do
  IFS=":" read -r name start_date end_date progress_artifact <<<"${shard}"
  window="sparse_${name}"
  if ! shard_selected "${name}" "${window}"; then
    continue
  fi
  matched_shards=$((matched_shards + 1))
  if ! handle_existing_window "${window}" "${start_date}" "${end_date}" "${progress_artifact}"; then
    continue
  fi
  run_cmd=(
    "${SCRIPT_DIR}/run_i12_pit_shard_supervised.sh"
    --schema "${SCHEMA}"
    --source-hur-schema "${SOURCE_HUR_SCHEMA}"
    --start-date "${start_date}"
    --end-date "${end_date}"
    --decision-time "${DECISION_TIME}"
    --minute-path-mode "${MINUTE_PATH_MODE}"
    --feed sip
    --intended-order-usd 250
    --max-spread-bps 200
    --max-quote-age-seconds 60
    --max-no-progress-minutes "${MAX_NO_PROGRESS_MINUTES}"
    --max-resumes "${MAX_RESUMES}"
    --artifact-base "${progress_artifact%.json}"
    --python-bin "${PYTHON_BIN}"
  )
  cmd="$(worker_shell_command "${run_cmd[@]}")"
  tmux new-window -d -t "${SESSION}" -n "${window}" -c "${ENGINE_DIR}" "${cmd}"
  launched_windows=$((launched_windows + 1))
  echo "launched ${SESSION}:${window} ${start_date}..${end_date}"
done

echo
echo "tmux windows:"
tmux list-windows -t "${SESSION}"
echo
echo "summary:"
echo "matched_shards=${matched_shards} launched_windows=${launched_windows} skipped_running_windows=${skipped_running_windows} replaced_stale_windows=${replaced_stale_windows} replaced_running_windows=${replaced_running_windows}"
echo "MAX_NO_PROGRESS_MINUTES=${MAX_NO_PROGRESS_MINUTES}"
echo "MAX_RESUMES=${MAX_RESUMES}"
echo
echo "process check:"
echo "tmux list-panes -a -t ${SESSION} -F '#S:#W #{pane_pid} #{pane_current_command}'"
echo "attach:"
echo "tmux attach -t ${SESSION}"
