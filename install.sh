#!/usr/bin/env bash
# Copyright (C) 2026 Neura-Tech-AI
# SPDX-License-Identifier: LGPL-3.0-only
#
# mixtureforge installer — the counterpart of `pip install mergekit`.
#
#   ./install.sh              # editable install into ./.venv  (development)
#   ./install.sh --user       # editable install into the current environment
#   ./install.sh --wheel      # build a wheel + install it (production)
#   ./install.sh --clean      # throw everything away and start over
#
# Everything it does is plain pip; the script exists so the steps are the same
# on every machine and so the "you already have upstream mergekit installed"
# trap is handled for you.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="venv"
CLEAN=0
PY="${PYTHON:-python3}"

for arg in "$@"; do
    case "$arg" in
        --user)     MODE="user" ;;
        --wheel)    MODE="wheel" ;;
        --venv)     MODE="venv" ;;
        --clean)    CLEAN=1 ;;
        -h|--help)
            sed -n '5,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown option: $arg (try --help)" >&2
            exit 2 ;;
    esac
done

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- python check
say "==> python"
command -v "$PY" >/dev/null 2>&1 || die "python3 not found"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "need Python >= 3.10, found $PYVER"
echo "    $PY ($PYVER)"

# ------------------------------------------------------- stale artefacts first
say "==> cleaning stale build artefacts"
rm -rf build dist ./*.egg-info
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

if [ "$CLEAN" = "1" ]; then
    say "==> --clean: removing .venv"
    rm -rf .venv
fi

# ------------------------------------------------------------------- venv mode
if [ "$MODE" = "venv" ]; then
    if [ ! -x .venv/bin/python ]; then
        say "==> creating .venv"
        "$PY" -m venv .venv
    fi
    PIP=".venv/bin/pip"
    PY_BIN=".venv/bin/python"
else
    PIP="$PY -m pip"
    PY_BIN="$PY"
fi

# A fork that reuses the upstream import name would collide. We renamed the
# package, but an old `mergekit` install still shadows nothing useful and
# confuses `pip list`, so offer to remove it.
say "==> checking for a conflicting upstream install"
if $PIP show mergekit >/dev/null 2>&1; then
    warn "    found an installed 'mergekit' package."
    warn "    mixtureforge now uses its own import name, so both can coexist,"
    warn "    but the old CLI names (mergekit-moe, ...) would still point at it."
    warn "    remove it with:  $PIP uninstall -y mergekit"
fi

# ------------------------------------------------------------------- install
case "$MODE" in
    venv|user)
        say "==> installing mixtureforge (editable)"
        $PIP install -q --upgrade pip >/dev/null 2>&1 || true
        $PIP install -e ".[dev]" || die "install failed"

        say "==> installing test extras"
        $PIP install -q pytest || die "pytest install failed"
        ;;
    wheel)
        # Always build and install through a virtualenv. System pythons are
        # frequently "externally managed" (PEP 668) and pip refuses to touch
        # them, which is exactly the trap this mode must not fall into.
        if [ ! -x .venv/bin/python ]; then
            say "==> creating .venv for the wheel build"
            "$PY" -m venv .venv
        fi
        say "==> building wheel"
        .venv/bin/pip install -q build || die "could not install 'build'"
        .venv/bin/python -m build --wheel || die "wheel build failed"
        WHEEL="$(ls -1 dist/mixtureforge-*.whl 2>/dev/null | head -1)"
        [ -n "$WHEEL" ] || die "no wheel produced"
        say "==> installing $WHEEL"
        .venv/bin/pip install --force-reinstall --no-deps "$WHEEL" \
            || die "wheel install failed"
        say "==> installing runtime dependencies"
        .venv/bin/pip install -q -e . || die "dependency install failed"
        PIP=".venv/bin/pip"
        PY_BIN=".venv/bin/python"
        ;;
esac

# ------------------------------------------------------------------- verify
say "==> verifying"
$PY_BIN - <<'PYEOF' || die "self-check failed"
import sys
import mixtureforge
from mixtureforge.architecture.json_definitions import NAME_TO_ARCH
from mixtureforge.moe import ALL_OUTPUT_ARCHITECTURES

assert mixtureforge.__file__, "package did not import"
assert len(NAME_TO_ARCH) >= 60, f"only {len(NAME_TO_ARCH)} architectures found"
assert any(a.name() == "Qwen3.5 MoE" for a in ALL_OUTPUT_ARCHITECTURES), "Qwen3.5 MoE missing"

print(f"    mixtureforge {getattr(mixtureforge, '__version__', '0.1.5')}")
print(f"    {len(NAME_TO_ARCH)} architectures, "
      f"{len(ALL_OUTPUT_ARCHITECTURES)} MoE outputs")
print("    import ok")
PYEOF

# the console scripts must be on PATH
BIN_DIR="$($PY_BIN -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
for cli in mixtureforge-moe mixtureforge-yaml; do
    if [ -x "$BIN_DIR/$cli" ]; then
        echo "    ok  $cli"
    else
        die "missing console script: $cli (looked in $BIN_DIR)"
    fi
done

case "$MODE" in
    venv) BIN_HINT=".venv/bin/" ;;
    *)    BIN_HINT="$BIN_DIR/" ;;
esac

cat <<EOF

$(printf '\033[32m')done.$(printf '\033[0m')

  ${BIN_HINT}mixtureforge-moe <config.yml> <out_dir> --copy-tokenizer
  ${BIN_HINT}mixtureforge-yaml <config.yml> <out_dir>

  python -c "import mixtureforge; print(mixtureforge.__file__)"

EOF
