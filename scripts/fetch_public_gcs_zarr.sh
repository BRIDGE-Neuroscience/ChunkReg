#!/bin/bash
# Download a public GCS prefix (a zarr store, or one of its datasets) with curl only.
#
#     fetch_public_gcs_zarr.sh BUCKET PREFIX DEST [PARALLEL]
#
# For a machine with no gcloud/gsutil: objects are listed through the anonymous
# XML API (public buckets allow it), then fetched by PARALLEL (default 8)
# resumable curls into DEST with PREFIX stripped. Re-running resumes: a file
# whose size already matches the listing is skipped. Ends with a size check and
# exits 1 if anything is missing or short.
#
#     fetch_public_gcs_zarr.sh hip_ct_zarr_vector_03987646472fethdsvdvdfg \
#         zarr_vectors_test/BigBrain_registration_2026/BigBrain2015_coronal_20um.ome.zarr/0/ \
#         /data/BigBrain2015_coronal_20um.ome.zarr/0 8
set -uo pipefail
BUCKET=$1; PREFIX=$2; DEST=$3; PAR=${4:-8}
PREFIX=${PREFIX%/}/
mkdir -p "$DEST"
LIST="$DEST/.gcs_listing"

list_objects() {
    local marker="" xml
    while :; do
        xml=$(curl -s --retry 5 "https://storage.googleapis.com/$BUCKET?prefix=$PREFIX&max-keys=1000${marker:+&marker=$marker}")
        echo "$xml" | grep -o '<Key>[^<]*</Key><Generation>[^<]*</Generation><MetaGeneration>[^<]*</MetaGeneration><LastModified>[^<]*</LastModified><ETag>[^<]*</ETag><Size>[0-9]*</Size>' \
            | sed -E 's#<Key>([^<]*)</Key>.*<Size>([0-9]*)</Size>#\1 \2#'
        [ "$(echo "$xml" | grep -o '<IsTruncated>[a-z]*' | sed 's/<IsTruncated>//')" = "true" ] || break
        marker=$(echo "$xml" | grep -o '<NextMarker>[^<]*' | sed 's/<NextMarker>//')
    done
}

list_objects > "$LIST"
n=$(wc -l < "$LIST"); total=$(awk '{s+=$2} END {printf "%.1f", s/1e9}' "$LIST")
echo "$n objects, $total GB under gs://$BUCKET/$PREFIX -> $DEST  ($PAR parallel)"

fetch_one() {
    local key=$1 size=$2 rel out
    rel=${key#"$PREFIX"}; out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then return 0; fi
    curl -s -f --retry 5 --retry-delay 3 -C - -o "$out" "https://storage.googleapis.com/$BUCKET/$key" \
        || { echo "FAILED $key" >&2; return 1; }
}
export -f fetch_one; export PREFIX DEST BUCKET
xargs -P "$PAR" -n 2 bash -c 'fetch_one "$0" "$1"' < "$LIST"

bad=0
while read -r key size; do
    out="$DEST/${key#"$PREFIX"}"
    if [ ! -f "$out" ] || [ "$(stat -c %s "$out")" != "$size" ]; then echo "MISSING/SHORT $out"; bad=$((bad+1)); fi
done < "$LIST"
echo "verified: $((n-bad))/$n objects complete"
[ "$bad" -eq 0 ]
