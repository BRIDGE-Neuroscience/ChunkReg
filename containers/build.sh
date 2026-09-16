#!/bin/bash
# Build chunkreg.sif from the repo root, which is what chunkreg.def's %files
# paths are relative to. Needs internet (pip, git, HuggingFace) and either root
# or --fakeroot; pass extra flags through, e.g.
#     bash containers/build.sh --fakeroot
#     bash containers/build.sh --fakeroot /SAN/.../images/chunkreg.sif
# The build copies the working tree, so uncommitted edits go into the image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

flags=()
out=chunkreg.sif
for arg in "$@"; do
    case "$arg" in
        -*) flags+=("$arg") ;;
        *)  out="$arg" ;;
    esac
done

RUNTIME=$(command -v apptainer || command -v singularity) \
    || { echo "no apptainer/singularity on PATH (try: module load apptainer)" >&2; exit 1; }

# Layers and the compile are large; keep them off a small /tmp or \$HOME.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$APPTAINER_TMPDIR/apptainer-cache}"

"$RUNTIME" build "${flags[@]}" "$out" containers/chunkreg.def
echo "built $out"
