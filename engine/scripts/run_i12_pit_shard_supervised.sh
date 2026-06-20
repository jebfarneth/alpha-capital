#!/usr/bin/env bash
set -euo pipefail

SCHEMA="${SCHEMA:-}"
SOURCE_HUR_SCHEMA="${SOURCE_HUR_SCHEMA:-public}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
DECISION_TIME="${DECISION_TIME:-09:40}"
MINUTE_PATH_MODE="${MINUTE_PATH_MODE:-strict_contiguous}"
ARTIFACT_BASE="${ARTIFACT_BASE:-}"
MAX_RESUMES="${MAX_RESUMES:-5}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FEED="${FEED:-sip}"
INTENDED_ORDER_USD="${INTENDED_ORDER_USD:-250}"
MAX_SPREAD_BPS="${MAX_SPREAD_BPS:-200}"
MAX_QUOTE_AGE_SECONDS="${MAX_QUOTE_AGE_SECONDS:-60}"
MAX_NO_PROGRESS_MINUTES="${MAX_NO_PROGRESS_MINUTES:-20}"

usage() {
  cat >&2 <<'EOF'
Usage: run_i12_pit_shard_supervised.sh --schema SCHEMA --start-date YYYY-MM-DD --end-date YYYY-MM-DD --artifact-base PATH [options]

Options:
  --schema SCHEMA
  --source-hur-schema SCHEMA        default: public
  --start-date YYYY-MM-DD
  --end-date YYYY-MM-DD
  --decision-time HH:MM             default: 09:40
  --minute-path-mode MODE           default: strict_contiguous
  --artifact-base PATH              writes PATH_attemptN.json
  --max-resumes N                   default: 5
  --python-bin PATH                 default: .venv/bin/python
  --feed FEED                       default: sip
  --intended-order-usd USD          default: 250
  --max-spread-bps BPS              default: 200
  --max-quote-age-seconds SECONDS   default: 60
  --max-no-progress-minutes MINUTES default: 20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --schema)
      SCHEMA="$2"
      shift 2
      ;;
    --source-hur-schema)
      SOURCE_HUR_SCHEMA="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --decision-time)
      DECISION_TIME="$2"
      shift 2
      ;;
    --minute-path-mode)
      MINUTE_PATH_MODE="$2"
      shift 2
      ;;
    --artifact-base)
      ARTIFACT_BASE="$2"
      shift 2
      ;;
    --max-resumes)
      MAX_RESUMES="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --feed)
      FEED="$2"
      shift 2
      ;;
    --intended-order-usd)
      INTENDED_ORDER_USD="$2"
      shift 2
      ;;
    --max-spread-bps)
      MAX_SPREAD_BPS="$2"
      shift 2
      ;;
    --max-quote-age-seconds)
      MAX_QUOTE_AGE_SECONDS="$2"
      shift 2
      ;;
    --max-no-progress-minutes)
      MAX_NO_PROGRESS_MINUTES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${SCHEMA}" || -z "${START_DATE}" || -z "${END_DATE}" || -z "${ARTIFACT_BASE}" ]]; then
  echo "ERROR: --schema, --start-date, --end-date, and --artifact-base are required" >&2
  usage
  exit 2
fi

if [[ "${MAX_RESUMES}" =~ [^0-9] || -z "${MAX_RESUMES}" ]]; then
  echo "ERROR: --max-resumes must be a non-negative integer" >&2
  exit 2
fi

parse_attempt_artifact() {
  local artifact="$1"
  python3 - "$artifact" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("\t")
    raise SystemExit(0)

data = json.loads(path.read_text())
event = data.get("event") or ""
last_progress_event = data.get("last_progress_event") or ""
last_payload = data.get("last_progress_payload") or {}
last_completed = data.get("last_completed_trading_date")
if not last_completed and event in {"date_finish", "date_finish_minimal", "finish"}:
    last_completed = data.get("last_trading_date")
if not last_completed:
    last_completed = last_payload.get("last_completed_trading_date")
if (
    not last_completed
    and last_progress_event in {"date_finish", "date_finish_minimal"}
):
    last_completed = last_payload.get("last_trading_date")
print(f"{event}\t{last_completed or ''}")
PY
}

next_calendar_day() {
  local last_date="$1"
  python3 - "$last_date" <<'PY'
from datetime import date, timedelta
import sys

print((date.fromisoformat(sys.argv[1]) + timedelta(days=1)).isoformat())
PY
}

date_on_or_after() {
  local value="$1"
  local threshold="$2"
  python3 - "$value" "$threshold" <<'PY'
from datetime import date
import sys

raise SystemExit(
    0
    if date.fromisoformat(sys.argv[1]) >= date.fromisoformat(sys.argv[2])
    else 1
)
PY
}

attempt=1
resumes_used=0
cur_start="${START_DATE}"

while true; do
  artifact="${ARTIFACT_BASE}_attempt${attempt}.json"
  run_cmd=(
    "${PYTHON_BIN}" -m alpha.jobs.run_i12_pit_rebuild
    --schema "${SCHEMA}"
    --source-hur-schema "${SOURCE_HUR_SCHEMA}"
    --start-date "${cur_start}"
    --end-date "${END_DATE}"
    --decision-time "${DECISION_TIME}"
    --minute-path-mode "${MINUTE_PATH_MODE}"
    --feed "${FEED}"
    --intended-order-usd "${INTENDED_ORDER_USD}"
    --max-spread-bps "${MAX_SPREAD_BPS}"
    --max-quote-age-seconds "${MAX_QUOTE_AGE_SECONDS}"
    --max-no-progress-minutes "${MAX_NO_PROGRESS_MINUTES}"
    --skip-final-report
    --progress-artifact "${artifact}"
  )

  echo "I12 PIT supervised attempt ${attempt}: ${cur_start}..${END_DATE} -> ${artifact}"
  set +e
  "${run_cmd[@]}"
  exit_code=$?
  set -e

  IFS=$'\t' read -r event last_completed_date < <(parse_attempt_artifact "${artifact}")
  if [[ "${exit_code}" -eq 0 && "${event}" == "finish" ]]; then
    echo "I12 PIT supervised shard finished after ${attempt} attempt(s)"
    exit 0
  fi
  if [[ -n "${last_completed_date}" ]] && date_on_or_after "${last_completed_date}" "${END_DATE}"; then
    echo "I12 PIT supervised shard reached ${END_DATE} before event='${event}' exit_code=${exit_code}; treating as complete"
    exit 0
  fi

  if [[ "${exit_code}" -eq 0 && "${event}" != "no_progress_timeout" ]]; then
    echo "ERROR: worker exited 0 but did not write finish event: event='${event}' artifact='${artifact}'" >&2
    exit 1
  fi

  if [[ "${event}" != "no_progress_timeout" && "${exit_code}" -ne 70 ]]; then
    echo "ERROR: worker failed with non-recoverable exit_code=${exit_code} event='${event}' artifact='${artifact}'" >&2
    exit "${exit_code}"
  fi

  if [[ "${resumes_used}" -ge "${MAX_RESUMES}" ]]; then
    echo "ERROR: shard did not finish after ${attempt} attempt(s); max resumes ${MAX_RESUMES} exhausted" >&2
    echo "last exit_code=${exit_code} event='${event}' artifact='${artifact}'" >&2
    if [[ "${exit_code}" -ne 0 ]]; then
      exit "${exit_code}"
    fi
    exit 1
  fi

  if [[ -n "${last_completed_date}" ]]; then
    cur_start="$(next_calendar_day "${last_completed_date}")"
  else
    echo "no completed date found in ${artifact}; retrying from ${cur_start}" >&2
  fi
  resumes_used=$((resumes_used + 1))
  attempt=$((attempt + 1))
  echo "resuming from ${cur_start} after event='${event}' exit_code=${exit_code}"
done
