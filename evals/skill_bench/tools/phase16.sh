#!/usr/bin/env bash
# The supervisors of the phase16 studies' steps: one command per step, with the same
# options every time, so a relaunch or the recovery after a reboot cannot drift from
# them. It works in the checkout it lives in, from anywhere.
#
#   phase16.sh print STEP [-- EXTRA...]    the supervisor's command line (changes nothing)
#   phase16.sh exec STEP [-- EXTRA...]     run the supervisor in the foreground (the
#                                          systemd unit skill-bench@STEP runs this)
#   phase16.sh up STEP... [-- EXTRA...]    start each step's supervisor in its tmux session
#                                          p16-STEP, unless a supervisor holds the runs
#                                          directory; a leftover session whose
#                                          supervisor exited is replaced
#   phase16.sh status [STEP...]            supervise --status of each step (default: all)
#   phase16.sh recover STEP...             after a reboot: the compose Postgres and the
#                                          WebShop server (both idempotent), then up STEP...
#   phase16.sh install-unit                write the systemd user unit skill-bench@.service
#                                          for this checkout (from the template
#                                          infra/skill-bench@.service.in)
#
# STEP: dev (phase16-dev), r1, r2, r3 (the three replicates
# of phase16). EXTRA goes to the supervisor of that launch only (the one-off options,
# e.g. -- --new-attempt-cells ... --new-attempt-note '...', or -- --widen-selection
# '...': the supervisor records them and leaves them out of every later restart).
#
# P16_METHODS and P16_BENCHMARKS narrow r1-r3 to some methods or benchmarks on every
# launch path (up, recover, exec and so the systemd unit), so the cells of a replicate
# can run in more than one launch into the same runs directories. The scheduler refuses
# a selection wider than every earlier launch of the runs directory unless
# -- --widen-selection 'NOTE' is given once (_schedule/selection.json records it).
#
# One budget at a time: every step's budget is a key-basis budget, which counts ALL
# spend on the key since its start, whoever spent it. So exec and up refuse to start a
# step while a supervisor of another budget runs, and up refuses steps of two budgets in
# one launch: the development pass and the replicates (r1-r3 share one budget
# start, so they run together) go one after the other. P16_ALLOW_OVERLAP=1 starts it
# anyway, knowing that each will count the other's spend. Spend on the key outside these
# steps (a cell run by hand, say) counts against them all alike.
#
# Settings come from the environment and from the launch config
# ${P16_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env} when it exists.
# The file is read after the environment, so a setting in both takes the file's value
# (the systemd unit and a cron line read the same file; it holds the key file's path,
# never a key). A step's budget and the key file are needed only where that step's
# supervisor is printed or started (print, exec, and up or recover for each step they
# start); status needs neither, and recover brings back Postgres and WebShop without them:
#   SKILL_BENCH_ENV_FILE   the key file, outside the checkout (required there)
#   P16_BUDGET             r1-r3: the USD budget for r1-r3 (required there)
#   P16_DEV_BUDGET         dev: the USD budget for dev (required there)
#   P16_METHODS            r1-r3: the methods of this launch, e.g. "none reasoningbank
#                          awm" (default: the study's four)
#   P16_BENCHMARKS         r1-r3: the benchmarks of this launch, e.g. "alfworld" (default:
#                          the study's, ALFWorld and WebShop)
#   P16_CONCURRENCY        dev, r1-r3: --concurrency (4; 3 eases a loaded host)
#   P16_NOTIFY_CMD         a --notify-cmd for every supervisor (optional)
#   P16_GATES              default: --min-free-memory-gb 8 --min-free-disk-gb 15
#                          --memory-emergency-gb 3
#   P16_PORT_BASE          the gateway ports: rN gets base+N, dev base+5
#                          (default 18540)
#   ALFWORLD_DATA          default: $HOME/.cache/alfworld
#   P16_PYTHON             default: .venv/bin/python (in the checkout)
#   P16_RUNS               default: evals/skill_bench/runs
#   P16_WEBSHOP_WAIT_S     recover: how long to wait for the WebShop server (1800)
#   P16_ALLOW_OVERLAP      1: start a step although another budget's supervisor runs
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root" || exit 1
config=${P16_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env}
if [ -r "$config" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$config"
  set +a
fi
PYTHON=${P16_PYTHON:-.venv/bin/python}
P16_RUNS=${P16_RUNS:-evals/skill_bench/runs}
STEPS="dev r1 r2 r3"
UNIT_TEMPLATE=evals/skill_bench/infra/skill-bench@.service.in

die() {
  echo "phase16.sh: $*" >&2
  exit 2
}

# The header above (the usage), without the shebang.
usage() {
  awk 'NR == 1 { next } /^#/ { print; next } { exit }' "$here/phase16.sh" >&2
}

# STEP's runs directory.
runs_of() {
  case $1 in
    dev | r1 | r2 | r3) echo "$P16_RUNS/phase16-$1" ;;
    *) die "unknown step '$1' (one of: $STEPS)" ;;
  esac
}

# STEP's budget: the replicates share one budget start; dev has its own.
budget_of() {
  case $1 in
    r1 | r2 | r3) echo replicates ;;
    *) echo "$1" ;;
  esac
}

# The supervisor's options of STEP into ARGS and its runs directory into RUNS; r1-r3 take
# their method selection from P16_METHODS and their benchmark selection from
# P16_BENCHMARKS.
step_args() {
  local step=$1 study runs budget port concurrency gates base
  local -a methods=() benchmarks=()
  runs=$(runs_of "$step") || exit 2
  base=${P16_PORT_BASE:-18540}
  case $base in
    '' | *[!0-9]*) die "P16_PORT_BASE must be a number (default 18540), not '$base'" ;;
  esac
  case $step in
    dev)
      study=phase16-dev budget=${P16_DEV_BUDGET:-} port=$((10#$base + 5)) concurrency=${P16_CONCURRENCY:-4}
      [ -n "$budget" ] || die "set P16_DEV_BUDGET, the USD budget for dev" ;;
    r1 | r2 | r3)
      study=phase16 budget=${P16_BUDGET:-} port=$((10#$base + ${step#r})) concurrency=${P16_CONCURRENCY:-4}
      [ -n "$budget" ] || die "set P16_BUDGET, the USD budget for r1-r3"
      read -r -a methods <<<"${P16_METHODS:-}"
      read -r -a benchmarks <<<"${P16_BENCHMARKS:-}" ;;
  esac
  [ -n "${SKILL_BENCH_ENV_FILE:-}" ] || die "set SKILL_BENCH_ENV_FILE (the key file's path)"
  ARGS=(-m evals.skill_bench.tools.supervise --runs-dir "$runs" --study "$study")
  if [ ${#methods[@]} -gt 0 ]; then
    ARGS+=(--method "${methods[@]}")
  fi
  if [ ${#benchmarks[@]} -gt 0 ]; then
    ARGS+=(--benchmark "${benchmarks[@]}")
  fi
  ARGS+=(--concurrency "$concurrency" --test-concurrency 4 --budget "$budget"
    --balance-floor 10 --env-file "$SKILL_BENCH_ENV_FILE" --gateway-port "$port")
  case $step in
    r2 | r3) ARGS+=(--budget-start-from "$P16_RUNS/phase16-r1") ;;
  esac
  read -r -a gates <<<"${P16_GATES:---min-free-memory-gb 8 --min-free-disk-gb 15 --memory-emergency-gb 3}"
  ARGS+=("${gates[@]}")
  if [ -n "${P16_NOTIFY_CMD:-}" ]; then
    ARGS+=(--notify-cmd "$P16_NOTIFY_CMD")
  fi
  RUNS=$runs
}

# The steps whose runs directory a live supervisor holds (its flock), into HELD.
read_held() {
  local step pairs=()
  for step in $STEPS; do
    pairs+=("$step" "$(runs_of "$step")")
  done
  HELD=$("$PYTHON" -c 'import sys
from pathlib import Path
from evals.skill_bench.infra.processes import lock_is_held
pairs = sys.argv[1:]
for step, runs in zip(pairs[::2], pairs[1::2]):
    if lock_is_held(Path(runs) / "_supervise" / "supervisor.lock"):
        print(step)' "${pairs[@]}")
}

# Whether STEP is in HELD.
held() {
  # shellcheck disable=SC2086 # word splitting joins HELD's lines
  case " $(echo $HELD) " in *" $1 "*) return 0 ;; esac
  return 1
}

# Refuse STEP while a supervisor of another budget runs (see "One budget at a time").
one_budget_at_a_time() {
  local step=$1 other
  [ "${P16_ALLOW_OVERLAP:-}" = 1 ] && return 0
  for other in $HELD; do
    [ "$(budget_of "$other")" = "$(budget_of "$step")" ] && continue
    die "$step: the $other supervisor runs ($(runs_of "$other")), and a key-basis budget counts every spend on the key: each would use up the other's budget. Wait until it exits 0 (phase16.sh status $other), or set P16_ALLOW_OVERLAP=1 to start $step anyway"
  done
}

# Refuse one launch of steps of different budgets (see "One budget at a time").
one_budget_per_launch() {
  local step
  [ "${P16_ALLOW_OVERLAP:-}" = 1 ] && return 0
  for step in "$@"; do
    [ "$(budget_of "$step")" = "$(budget_of "$1")" ] ||
      die "$1 and $step have budgets of their own, and a key-basis budget counts every spend on the key: start one, and the other once it exits 0, or set P16_ALLOW_OVERLAP=1 to start both"
  done
}

# Refuse up and recover without tmux, which holds each supervisor's session.
need_tmux() {
  command -v tmux >/dev/null 2>&1 ||
    die "tmux is not installed (not on PATH), and up starts each supervisor in a tmux session: install tmux, or run the step with phase16.sh exec STEP (in the foreground, or as the systemd unit skill-bench@STEP)"
}

# STEP... [-- EXTRA...] into STEP_LIST and EXTRA.
split_args() {
  STEP_LIST=()
  EXTRA=()
  while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
      shift
      EXTRA=("$@")
      return
    fi
    STEP_LIST+=("$1")
    shift
  done
}

up() {
  local step session command
  [ ${#STEP_LIST[@]} -gt 0 ] || die "name the steps to start (one or more of: $STEPS)"
  for step in "${STEP_LIST[@]}"; do
    runs_of "$step" >/dev/null || exit 2
  done
  one_budget_per_launch "${STEP_LIST[@]}"
  read_held
  for step in "${STEP_LIST[@]}"; do
    one_budget_at_a_time "$step"
  done
  # The settings (a budget among them) of every step this launch starts, before any start.
  for step in "${STEP_LIST[@]}"; do
    held "$step" || step_args "$step"
  done
  need_tmux
  for step in "${STEP_LIST[@]}"; do
    if held "$step"; then
      echo "$step: a supervisor holds $(runs_of "$step"); left as it is"
      continue
    fi
    step_args "$step"
    session=p16-$step
    if tmux has-session -t "=$session" 2>/dev/null; then
      tmux kill-session -t "=$session"  # its supervisor exited (no lock held)
    fi
    command=$(printf '%q ' "$here/phase16.sh" exec "$step" -- ${EXTRA[@]+"${EXTRA[@]}"})
    command=${command% }
    tmux new-session -d -s "$session" "$command; echo \"supervise exited with \$?\"; exec bash" ||
      die "$step: tmux could not start the session $session (exit $?); its supervisor is not running"
    echo "$step: started its supervisor in tmux session $session"
  done
}

# Write the systemd user unit for this checkout: the template with @CHECKOUT@ replaced by
# the checkout's path, into the user's unit directory (systemd reads
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user).
install_unit() {
  local text unit_dir unit
  [ -r "$UNIT_TEMPLATE" ] || die "no unit template at $root/$UNIT_TEMPLATE"
  case $root in
    *[[:space:]%\\\"\']*)
      die "the checkout's path ($root) holds a space, a quote, a backslash or %, which the unit's ExecStart cannot take as it is: move the checkout" ;;
  esac
  text=$(cat "$UNIT_TEMPLATE") || die "cannot read $root/$UNIT_TEMPLATE"
  text=${text//@CHECKOUT@/"$root"}
  unit_dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
  unit=$unit_dir/skill-bench@.service
  mkdir -p "$unit_dir" || die "cannot create $unit_dir"
  printf '%s\n' "$text" >"$unit" || die "cannot write $unit"
  echo "wrote $unit for the checkout $root; then:"
  echo "  systemctl --user daemon-reload"
  echo "  systemctl --user enable --now skill-bench@r1 skill-bench@r2 skill-bench@r3"
  echo "  sudo loginctl enable-linger \"\$USER\"    # user units start at boot, without a login"
}

command=${1:-}
[ $# -gt 0 ] && shift
case $command in
  print | exec)
    split_args "$@"
    [ ${#STEP_LIST[@]} -eq 1 ] || die "$command takes one step (one of: $STEPS)"
    step_args "${STEP_LIST[0]}"
    if [ "$command" = print ]; then
      printf '%q ' "$PYTHON" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
      echo
      exit 0
    fi
    read_held
    one_budget_at_a_time "${STEP_LIST[0]}"
    export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
    exec "$PYTHON" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  up)
    split_args "$@"
    up
    ;;
  status)
    code=0
    for step in ${*:-$STEPS}; do
      RUNS=$(runs_of "$step") || exit 2
      echo "== $step ($RUNS)"
      "$PYTHON" -m evals.skill_bench.tools.supervise --status --runs-dir "$RUNS" || code=1
    done
    exit $code
    ;;
  recover)
    split_args "$@"
    [ ${#STEP_LIST[@]} -gt 0 ] || die "name the steps to start (one or more of: $STEPS)"
    need_tmux
    docker compose -f evals/skill_bench/infra/compose.yaml up -d ||
      echo "phase16.sh: the compose Postgres did not start; the MIRIX cells wait" >&2
    "$PYTHON" -m evals.skill_bench.envs.webshop --start --wait-s "${P16_WEBSHOP_WAIT_S:-1800}" ||
      echo "phase16.sh: the WebShop server is not ready; the WebShop cells wait in its gate" >&2
    up
    ;;
  install-unit)
    [ $# -eq 0 ] || die "install-unit takes no arguments"
    install_unit
    ;;
  *)
    usage
    exit 2
    ;;
esac
