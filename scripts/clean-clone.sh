#!/usr/bin/env bash
# The clean-clone check: put the repository in a directory of its own and run
# the quick start there, with nothing on the machine that tree did not bring --
# no `build/`, no virtual environment, a fresh uv cache and a fresh managed
# Python, so every package comes from the index and the checkpoint from the
# Hub.  The run is `make demo`, the command the front page gives: environment,
# checkpoint, quantize, compile, harness build, and the model on qcore_top.
#
# There are two trees a run can check, and it names the one it checked in its
# first line, in its last line and in the report:
#
#   the committed tree     the default: `git clone` of the source at HEAD, or
#                          at --ref, which is what anyone else can fetch;
#   the tree as it stands  --worktree: every path git tracks, at the content on
#                          disk, staged and unstaged changes included -- the
#                          tree a commit made right now would carry.
#
# Neither carries a file that was never added, so a file left untracked stops
# the demo here rather than on a reader's machine.
#
# The demo's own verdict decides this one: the ids the hardware generated
# against `models/<name>/expected_tokens.json` and the counters a clean run
# leaves at zero.  Every stage carries its wall clock, and the report goes to
# `build/clean-clone/<model>.json` in the repository the script was run from.
#
# A run that stops names what stopped it: which stage of this script, which
# stage of the demo under it, the seconds it had run for and the last lines it
# printed.  The report is written either way, and the demo's whole output is
# kept beside it as `build/clean-clone/<model>.log`, so the clone can go and
# the evidence stays.
#
#   scripts/clean-clone.sh                 SmolLM2-135M-Instruct, the quick start's first line
#   scripts/clean-clone.sh --model qwen    its second line
#   scripts/clean-clone.sh --worktree      the tree as it stands, not the last commit
#   scripts/clean-clone.sh --keep          leave the clone where it ran
#   scripts/clean-clone.sh --ref v0.1.0    clone a tag instead of HEAD
set -euo pipefail
export LC_ALL=C

REPO=$(cd "$(dirname "$0")/.." && pwd -P)

MODEL=smollm2
SOURCE=$REPO
REF=
DEST=
KEEP=0
MAX_NEW=20
REPORT=
TREE=committed
REF_GIVEN=

usage() {
  cat <<'USAGE'
usage: scripts/clean-clone.sh [options]

  --model ALIAS    qwen, smollm2 or a Hugging Face repo id (smollm2)
  --source PATH    the repository to take the tree from (this one)
  --ref REV        the revision to check out (the source's HEAD)
  --worktree       check the tree as it stands -- every tracked path at the
                   content on disk -- instead of the committed tree
  --dir PATH       where to clone (a fresh directory under TMPDIR)
  --max-new N      decode steps; the run stops at an end-of-sequence id (20)
  --report PATH    where the JSON report goes (build/clean-clone/<model>.json)
  --keep           leave the clone, its uv cache and its build outputs in place
USAGE
}

while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --ref) REF=$2; REF_GIVEN=1; shift 2 ;;
    --dir) DEST=$2; shift 2 ;;
    --max-new) MAX_NEW=$2; shift 2 ;;
    --report) REPORT=$2; shift 2 ;;
    --worktree) TREE=working; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "clean-clone: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

SLUG=$(printf '%s' "$MODEL" | tr 'A-Z/' 'a-z-')
[ -n "$REPORT" ] || REPORT="$REPO/build/clean-clone/$SLUG.json"

# What the quick start needs on PATH.  Verilator drives a C++ compiler and the
# demo needs nothing else: Yosys and Icarus belong to `make lint`, `make synth`
# and `make gatesim`.
CXX_BIN=
for c in c++ g++ clang++; do
  if command -v "$c" > /dev/null 2>&1; then CXX_BIN=$c; break; fi
done
MISSING=
for t in git uv verilator make; do
  command -v "$t" > /dev/null 2>&1 || MISSING="$MISSING $t"
done
[ -n "$CXX_BIN" ] || MISSING="$MISSING c++"
if [ -n "$MISSING" ]; then
  echo "clean-clone: not on PATH:$MISSING" >&2
  echo "clean-clone: the quick start needs git, uv, Verilator 5.x, make and a C++ compiler (README, Requirements)" >&2
  exit 2
fi

json_escape() { printf '%s' "$1" | sed 's/[\\"]/\\&/g'; }

# What stopped the run, filled in by `stage` and read by the report: the stage
# of this script that did not finish, and the status it exited with.  The
# seconds it had run for are on the stage table, which is printed either way.
FAILED_STAGE=
FAILED_RC=0
VERDICT=OK
WORK=
LOG=
STAGES=

first_line() { "$@" 2>/dev/null | sed -n 1p || true; }
V_GIT=$(first_line git --version)
V_UV=$(first_line uv --version)
V_VERILATOR=$(first_line verilator --version)
V_MAKE=$(first_line make --version)
V_CXX=$(first_line "$CXX_BIN" --version)

# What the source holds, and how far it is from its own HEAD: the CHANGED
# tracked files are in the tree as it stands, the UNTRACKED ones in neither
# tree.
SUBJECT=
CHANGED=0
UNTRACKED=0
if [ -d "$SOURCE" ]; then
  SOURCE=$(cd "$SOURCE" && pwd -P)
  if ! git -C "$SOURCE" rev-parse --git-dir > /dev/null 2>&1; then
    echo "clean-clone: $SOURCE is not a git repository, so it has no committed tree to clone" >&2
    echo "clean-clone: --source names the repository to take the tree from" >&2
    exit 2
  fi
  # Resolve what is to be checked before anything is built on it, so a revision
  # the source does not carry is said here rather than by git several steps on.
  WANT=${REF:-HEAD}
  REF=$(git -C "$SOURCE" rev-parse --verify --quiet "$WANT^{commit}" || true)
  if [ -z "$REF" ]; then
    echo "clean-clone: $SOURCE has no revision $WANT to check" >&2
    echo "clean-clone: --ref takes a commit, branch or tag of the source (git -C $SOURCE log --oneline -5 lists a few)" >&2
    exit 2
  fi
  SUBJECT=$(git -C "$SOURCE" log -1 --format=%s "$REF")
  UNTRACKED=$(git -C "$SOURCE" status --porcelain -uall | grep -c '^??' || true)
  CHANGED=$(( $(git -C "$SOURCE" status --porcelain -uall | wc -l) - UNTRACKED ))
elif [ "$TREE" = working ]; then
  echo "clean-clone: --worktree needs a source on this machine, not $SOURCE" >&2
  exit 2
fi

if [ "$TREE" = working ] && [ -n "$REF_GIVEN" ]; then
  echo "clean-clone: --ref names a commit, so it belongs to the committed tree, not --worktree" >&2
  exit 2
fi

SHORT=HEAD
[ -z "$REF" ] || SHORT=${REF:0:12}
if [ "$TREE" = working ]; then
  WHAT="the tree as it stands in $SOURCE"
  FROM="the tree on disk"
else
  WHAT="the committed tree at $SHORT"
  FROM="git clone"
fi

if [ -n "$DEST" ]; then
  mkdir -p "$DEST"
  WORK=$(cd "$DEST" && pwd -P)
  [ -z "$(ls -A "$WORK")" ] || { echo "clean-clone: --dir $WORK is not empty" >&2; exit 2; }
else
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/quettos-clean-clone.XXXXXX")
fi
case "$WORK/" in
  "$REPO"/*) echo "clean-clone: the clone goes outside the working tree, not $WORK" >&2; exit 2 ;;
esac

CLONE="$WORK/Quettos-open-accelerator"
LOG="$WORK/demo.log"
STAGES="$WORK/stages.jsonl"
: > "$STAGES"

cleanup() {
  [ "$KEEP" = 1 ] || rm -rf "$WORK"
}
trap cleanup EXIT

# Stage output goes to the terminal; only the `time` keyword's own line is
# captured, so every stage carries the seconds it took.
exec 3>&1 4>&2

stage() {
  local name=$1 note=$2
  shift 2
  local secs rc=0
  printf '\n=== %s: %s\n' "$name" "$note" >&3
  TIMEFORMAT=%R
  secs=$( { time "$@" 1>&3 2>&4; } 2>&1 ) || rc=$?
  printf '{"name": "%s", "seconds": %s, "note": "%s"}\n' "$name" "$secs" "$note" >> "$STAGES"
  if [ "$rc" != 0 ]; then FAILED_STAGE=$name; FAILED_RC=$rc; fi
  return $rc
}

# A CI checkout carries the commit under test at a detached HEAD, with a
# remote-tracking or a pull-request ref on it and no branch.  A local clone
# copies the whole object database, so the commit comes across either way; the
# check before `checkout` is there so that a source that somehow does not carry
# it is named here, rather than leaving git to say that a reference is not a
# tree.
clone() {
  git clone --quiet --no-checkout "$SOURCE" "$CLONE"
  if [ -n "$REF" ]; then
    if ! git -C "$CLONE" cat-file -e "$REF^{commit}" 2> /dev/null; then
      echo "clean-clone: the clone of $SOURCE carries no commit $REF" >&2
      return 1
    fi
    git -C "$CLONE" -c advice.detachedHead=false checkout --quiet --detach "$REF"
  else
    git -C "$CLONE" checkout --quiet
    REF=$(git -C "$CLONE" rev-parse HEAD)
    SUBJECT=$(git -C "$CLONE" log -1 --format=%s)
  fi
  git -C "$CLONE" --no-pager log -1 --format='%h %s'
}

# The tree as it stands: every path the index names, copied at the content on
# disk, so staged and unstaged changes are both in it and a file that was never
# added is not -- the tree a commit made right now would carry.  It reads the
# source and writes only into the destination.
export_worktree() {
  local f missing=0 n=0
  mkdir -p "$CLONE"
  while IFS= read -r -d '' f; do
    if [ ! -e "$SOURCE/$f" ]; then
      echo "clean-clone: $f is tracked and not on disk" >&2
      missing=$((missing + 1))
      continue
    fi
    mkdir -p "$CLONE/$(dirname "$f")"
    cp -Pp "$SOURCE/$f" "$CLONE/$f"
    n=$((n + 1))
  done < <(git -C "$SOURCE" ls-files -z)
  if [ "$missing" != 0 ]; then
    echo "clean-clone: $missing tracked file(s) are not on disk, so there is no tree to check" >&2
    return 1
  fi
  printf '%s tracked files at their content on disk, %s of them changed since %s\n' \
    "$n" "$CHANGED" "$SHORT"
}

run_demo() {
  (
    cd "$CLONE"
    UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/uv-python" \
      make demo MODEL="$MODEL" MAX_NEW="$MAX_NEW"
  ) 2>&1 | tee "$LOG"
}

# What the demo's stages left behind, and what the run printed while it was in
# one.  The demo prints `=== <stage>: <note>` as it enters a stage, so the last
# such line in its log names the stage that did not finish and everything after
# it is what that stage had to say.
DEMO_STAGES="$CLONE/build/demo/$SLUG/stages.jsonl"
IDS=
TEXT=
NAME=
CKPT=0
TOTAL_S=0.00
DEMO_STAGE=

inner() {
  [ ! -f "$STAGES" ] || sed -n 1p "$STAGES"
  [ ! -f "$DEMO_STAGES" ] || cat "$DEMO_STAGES"
}

demo_stage() {
  [ -f "$LOG" ] || return 0
  sed -n 's/^=== \([A-Za-z0-9_-]*\): .*/\1/p' "$LOG" | sed -n '$p'
}

demo_tail() {
  [ -f "$LOG" ] || return 0
  awk '/^=== [A-Za-z0-9_-]+: /{n = NR} {l[NR] = $0}
       END {for (i = (n ? n : 1); i <= NR; i++) print l[i]}' "$LOG" | tail -n "${1:-20}"
}

# The report, written whatever happened, and the demo's whole output kept beside
# it: the clone is about to go and the evidence has to outlive it.
write_report() {
  if [ -f "$LOG" ]; then
    IDS=$(sed -n 's/^ *ids  *\(\[.*\]\)$/\1/p' "$LOG" | sed -n 1p || true)
    TEXT=$(sed -n "s/^ *text  *'\(.*\)'$/\1/p" "$LOG" | sed -n 1p || true)
    DEMO_STAGE=$(demo_stage)
  fi
  NAME=$(ls -1 "$CLONE/build/models" 2> /dev/null | sed -n 1p || true)
  if [ -n "$NAME" ] && [ -f "$CLONE/build/models/$NAME/model.safetensors" ]; then
    CKPT=$(wc -c < "$CLONE/build/models/$NAME/model.safetensors" | tr -d ' ')
  fi
  TOTAL_S=$(sed -n 's/.*"seconds": \([0-9.]*\).*/\1/p' "$STAGES" 2> /dev/null |
    awk '{t += $1} END {printf "%.2f", t}')
  [ -n "$TOTAL_S" ] || TOTAL_S=0.00

  mkdir -p "$(dirname "$REPORT")"
  [ ! -f "$LOG" ] || cp "$LOG" "${REPORT%.json}.log"
  {
    printf '{\n'
    printf '  "model": "%s",\n  "source": "%s",\n  "tree": "%s",\n  "ref": "%s",\n  "subject": "%s",\n' \
      "$MODEL" "$SOURCE" "$TREE" "$REF" "$(json_escape "$SUBJECT")"
    printf '  "tracked_changes": %s,\n  "untracked_files": %s,\n' "$CHANGED" "$UNTRACKED"
    printf '  "checkpoint_bytes": %s,\n' "$CKPT"
    printf '  "tools": {"git": "%s", "uv": "%s", "verilator": "%s", "make": "%s", "cxx": "%s"},\n' \
      "$V_GIT" "$V_UV" "$V_VERILATOR" "$V_MAKE" "$V_CXX"
    printf '  "stages": [\n'
    inner | awk '{ s = $0; sub(/^\{/, "", s); sub(/\}$/, "", s);
                   printf "%s    {%s}", (NR > 1 ? ",\n" : ""), s } END { printf "\n" }'
    printf '  ],\n'
    printf '  "seconds_total": %s,\n' "$TOTAL_S"
    printf '  "stopped_in": "%s",\n  "demo_stopped_in": "%s",\n' "$FAILED_STAGE" "$DEMO_STAGE"
    printf '  "ids": %s,\n' "${IDS:-[]}"
    printf '  "text": "%s",\n' "$(json_escape "$TEXT")"
    printf '  "verdict": "%s"\n}\n' "$(json_escape "$VERDICT")"
  } > "$REPORT"
}

print_stages() {
  printf '\n  wall clock, on %s with nothing cached\n' "$WHAT"
  inner | sed -n 's/^{"name": "\([^"]*\)", "seconds": \([0-9.]*\), "note": "\(.*\)"}$/\1\t\2\t\3/p' |
    while IFS="$(printf '\t')" read -r n s note; do printf '    %-12s %8.2f s   %s\n' "$n" "$s" "$note"; done
  printf '    %s\n' '----------------------------'
  printf '    %-12s %8.2f s   %s\n' 'end to end' "$TOTAL_S" "$FROM to text on qcore_top"
}

# Every way out of the run goes through here, so there is always a report and
# the last line always says which stage stopped it and how far it had got.
finish() {
  VERDICT=$1
  write_report
  print_stages
  if [ "$VERDICT" = OK ]; then
    printf '\nclean-clone: OK -- %s on %s, %s s from %s to %s on qcore_top; report %s\n' \
      "${NAME:-$MODEL}" "$WHAT" "$TOTAL_S" "$FROM" "'$TEXT'" "${REPORT#"$REPO"/}"
    exit 0
  fi
  if [ -n "$DEMO_STAGE" ] && [ "$FAILED_STAGE" = demo ]; then
    printf '\nclean-clone: the demo stopped in its %s stage; its last lines were\n\n' "$DEMO_STAGE" >&2
    demo_tail 20 | sed 's/^/    /' >&2
    printf '\nclean-clone: the whole run is in %s\n' "${REPORT%.json}.log" >&2
  fi
  printf '\nclean-clone: %s -- on %s; the run is in %s\n' "$VERDICT" "$WHAT" "$REPORT" >&2
  exit "$2"
}

# Room for a first run.  Nothing here is cached, so the clone pays for the
# checkpoint, the quantized weights, the compiled image, a uv cache, a managed
# Python and a Verilator object directory all at once: 823 MiB for the small
# model on the machine `docs/PERFORMANCE.md` names, and a larger checkpoint
# more.  The floor is 2 GiB, and the free space is on the opening line either
# way -- a build that stops halfway through for want of space says only that a
# compiler could not write a file.
NEED_MB=2048
FREE_MB=$(df -Pk "$WORK" | awk 'NR == 2 {printf "%d", $4 / 1024}')

echo "clean-clone: $MODEL, the quick start on $WHAT"
echo "clean-clone: the clone goes in $WORK, ${FREE_MB:-unknown} MB free"
if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt "$NEED_MB" ]; then
  echo "clean-clone: a first run needs about $NEED_MB MB there and has $FREE_MB MB" >&2
  echo "clean-clone: TMPDIR chooses where the clone goes, and --dir names the directory outright" >&2
  finish "FAILED: $WORK has $FREE_MB MB free, short of the $NEED_MB MB a first run needs" 2
fi
if [ "$TREE" = working ]; then
  echo "clean-clone: the $CHANGED tracked file(s) changed since $SHORT are in it, and the $UNTRACKED untracked file(s) are not, as a commit would not carry them either"
  stage export "the tracked tree of $(basename "$SOURCE") at its content on disk" export_worktree ||
    finish "FAILED: the tree as it stands could not be exported (exit $FAILED_RC)" "$FAILED_RC"
else
  if [ "$CHANGED" != 0 ] || [ "$UNTRACKED" != 0 ]; then
    echo "clean-clone: the source's $CHANGED tracked change(s) and $UNTRACKED untracked file(s) are outside what this run checks; --worktree checks the tree as it stands"
  fi
  stage clone "git clone $(basename "$SOURCE") at $SHORT" clone ||
    finish "FAILED: $SHORT could not be cloned out of $SOURCE (exit $FAILED_RC)" "$FAILED_RC"
fi

# Nothing that tree did not bring: no build outputs, no environment, and, where
# the tree came from a clone, a working tree that matches the commit.
for p in build .venv; do
  if [ -e "$CLONE/$p" ]; then
    finish "FAILED: the clone carries $p, so it is not clean" 1
  fi
done
if [ "$TREE" = committed ] && [ -n "$(git -C "$CLONE" status --porcelain)" ]; then
  git -C "$CLONE" --no-pager status --short >&2
  finish "FAILED: the clone's working tree does not match $REF" 1
fi

rc=0
stage demo "make demo MODEL=$MODEL MAX_NEW=$MAX_NEW, fresh uv cache and managed Python" run_demo || rc=$?

if [ "$rc" != 0 ]; then
  finish "FAILED: make demo exited $rc" "$rc"
elif ! grep -q '^demo: OK' "$LOG"; then
  finish "FAILED: the demo printed no verdict of its own" 1
fi
finish OK 0
