#!/bin/bash
# Build chunkreg.sif on a Linux machine, for copying to the cluster.
#
#   bash containers/build.sh                     # CUDA 12.1, ./chunkreg.sif
#   bash containers/build.sh --cuda 12.4 --out /data/images/chunkreg.sif
#
# Picking --cuda: on a cluster GPU node (qrsh), run nvidia-smi and read the
# "Driver Version" in its header.
#   driver >= 560   any of 12.1, 12.4, 12.6
#   driver >= 550   12.1 or 12.4
#   driver >= 525   12.1   (the default)
# The cluster already runs torch 2.5.1 built for CUDA 12.0 on its H100s, so
# its driver is at least 525 and the default is safe there. A newer --cuda
# only buys a newer torch.
#
# Needs: Linux x86_64, apptainer (or singularity), internet access, ~30 GB free
# under the build tmp dir, and either root, sudo or --fakeroot. The build copies
# this working tree, uncommitted edits included, into the image.
set -euo pipefail

usage() {
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --cuda VER      12.1 (default), 12.4 or 12.6
  --out PATH      where to write the image (default: ./chunkreg.sif)
  --mode MODE     how to get root for the build: auto (default), root,
                  sudo or fakeroot
  --jobs N        parallel compile jobs for FireANTs' CUDA ops (default 4)
  --force         overwrite an existing image
  -h, --help      this text
EOF
}

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

CUDA=12.1
OUT=chunkreg.sif
MODE=auto
JOBS=4
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda)  CUDA="${2:?--cuda needs a value}"; shift 2 ;;
        --out)   OUT="${2:?--out needs a value}"; shift 2 ;;
        --mode)  MODE="${2:?--mode needs a value}"; shift 2 ;;
        --jobs)  JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

# One row per CUDA version: the devel base (nvcc is needed for the fused ops),
# the torch release, and the wheel index that matches both.
case "$CUDA" in
    12.1) BASE=nvidia/cuda:12.1.1-devel-ubuntu22.04 TORCH=2.5.1 TAG=cu121 ;;
    12.4) BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04 TORCH=2.5.1 TAG=cu124 ;;
    12.6) BASE=nvidia/cuda:12.6.3-devel-ubuntu22.04 TORCH=2.7.1 TAG=cu126 ;;
    *) die "--cuda must be 12.1, 12.4 or 12.6, not '$CUDA'" ;;
esac
INDEX="https://download.pytorch.org/whl/$TAG"

# ---- preflight -------------------------------------------------------------
say "checking this machine"

[ "$(uname -s)" = Linux ]   || die "this has to run on Linux (found $(uname -s))"
[ "$(uname -m)" = x86_64 ]  || die "the cluster is x86_64; building on $(uname -m) would not run there"

RUNTIME=$(command -v apptainer || command -v singularity || true)
[ -n "$RUNTIME" ] || die "no apptainer or singularity on PATH.
  Ubuntu:  sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer
  Others:  https://apptainer.org/docs/admin/main/installation.html"
echo "runtime  $("$RUNTIME" --version)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF="$REPO/containers/chunkreg.def"
for f in "$DEF" "$REPO/pyproject.toml" "$REPO/README.md" "$REPO/chunkreg/__init__.py"; do
    [ -e "$f" ] || die "missing $f; run this from a full chunkreg checkout"
done
# %files lines are whitespace separated, so a space in the path would split it.
case "$REPO" in *[[:space:]]*) die "the checkout path has a space in it ($REPO); move it" ;; esac
echo "source   $REPO"

case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
mkdir -p "$(dirname "$OUT")"
if [ -e "$OUT" ] && [ "$FORCE" != 1 ]; then
    die "$OUT already exists; pass --force to replace it"
fi
echo "output   $OUT"

# Layers and the compile are large; keep them off a small /tmp if asked to.
BUILD_TMP="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR:-${TMPDIR:-/tmp}}}"
mkdir -p "$BUILD_TMP"
free_gb=$(df -Pk "$BUILD_TMP" | awk 'NR==2 {print int($4/1048576)}')
echo "tmp      $BUILD_TMP (${free_gb} GB free)"
if [ "$free_gb" -lt 30 ]; then
    die "the build needs about 30 GB under $BUILD_TMP. Point it somewhere
  bigger:  APPTAINER_TMPDIR=/big/disk/tmp bash $0 ..."
fi
export APPTAINER_TMPDIR="$BUILD_TMP" SINGULARITY_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$BUILD_TMP/apptainer-cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

for url in https://pypi.org/simple/ "$INDEX/" https://huggingface.co https://github.com; do
    curl -fsS -o /dev/null --max-time 20 "$url" \
        || die "cannot reach $url; the build downloads from it"
done
echo "network  ok"

# How to get the root the build needs.
if [ "$MODE" = auto ]; then
    if [ "$(id -u)" = 0 ]; then
        MODE=root
    elif sudo -n true 2>/dev/null; then
        MODE=sudo
    else
        MODE=fakeroot
    fi
fi
case "$MODE" in
    root)     BUILD=("$RUNTIME" build) ;;
    # -E keeps the tmp and cache dirs chosen above.
    sudo)     BUILD=(sudo -E "$RUNTIME" build) ;;
    fakeroot) BUILD=("$RUNTIME" build --fakeroot) ;;
    *) die "--mode must be auto, root, sudo or fakeroot, not '$MODE'" ;;
esac
echo "mode     $MODE"

# ---- build -----------------------------------------------------------------
WORK="$(mktemp -d "$BUILD_TMP/chunkreg-def.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
sed -e "s|@BASE_IMAGE@|$BASE|g" \
    -e "s|@CUDA@|$CUDA|g" \
    -e "s|@TORCH_VERSION@|$TORCH|g" \
    -e "s|@TORCH_INDEX@|$INDEX|g" \
    -e "s|@MAX_JOBS@|$JOBS|g" \
    -e "s|@REPO@|$REPO|g" \
    "$DEF" > "$WORK/chunkreg.def"
if grep -n '@[A-Z_]*@' "$WORK/chunkreg.def"; then
    die "placeholders left unfilled in the definition (listed above)"
fi

say "building: CUDA $CUDA, torch $TORCH ($TAG), base $BASE"
echo "this takes 20-40 minutes, most of it compiling FireANTs' CUDA ops"
# Build to a temporary name so a failed build never leaves a broken image
# where a good one is expected.
PARTIAL="$OUT.partial"
rm -f "$PARTIAL"
if ! "${BUILD[@]}" "$PARTIAL" "$WORK/chunkreg.def"; then
    rm -f "$PARTIAL"
    if [ "$MODE" = fakeroot ]; then
        die "the build failed. If the error mentions fakeroot or subuid, this
  account cannot use --fakeroot here; rerun with --mode sudo, or ask an
  admin to run: sudo apptainer config fakeroot --add $(id -un)"
    fi
    die "the build failed; the error is above"
fi
# A sudo build leaves the image owned by root.
if [ "$MODE" = sudo ]; then
    sudo chown "$(id -u):$(id -g)" "$PARTIAL"
fi
mv -f "$PARTIAL" "$OUT"

# ---- report ----------------------------------------------------------------
# %test has already run inside the build; run the import check once more from
# the finished file, in case the move or a filesystem quirk damaged it.
say "checking the finished image"
"$RUNTIME" exec "$OUT" chunkreg --version

say "done"
echo "image    $OUT ($(du -h "$OUT" | cut -f1))"
echo "sha256   $(sha256sum "$OUT" | cut -d' ' -f1)"
cat <<EOF

Copy it to shared storage on the cluster and point SIF at it in
sge/chunkreg.qsub, for example:

  rsync -P "$OUT" USER@CLUSTER:/home/USER/storage_main/images/

Then check it on a GPU node (qrsh) before submitting:

  apptainer exec --nv /home/USER/storage_main/images/$(basename "$OUT") \\
      python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
EOF
