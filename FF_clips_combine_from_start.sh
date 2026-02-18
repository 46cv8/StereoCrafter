#!/usr/bin/env bash
set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it first (e.g. sudo apt install -y ffmpeg)."
  exit 1
fi

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <start_clip_number> <stop_clip_number> <clips_folder>"
  echo "Example: $0 3 25 /path/to/Depth_001/stereopilot"
  exit 1
fi

START_RAW="$1"
STOP_RAW="$2"
CLIPS_DIR="$3"

if [[ ! "$START_RAW" =~ ^[0-9]+$ ]]; then
  echo "start_clip_number must be a non-negative integer. Got: $START_RAW"
  exit 1
fi
START_CLIP=$((10#$START_RAW))

if [[ ! "$STOP_RAW" =~ ^[0-9]+$ ]]; then
  echo "stop_clip_number must be a non-negative integer. Got: $STOP_RAW"
  exit 1
fi
STOP_CLIP=$((10#$STOP_RAW))

if (( STOP_CLIP < START_CLIP )); then
  echo "stop_clip_number must be >= start_clip_number. Got: start=$START_CLIP stop=$STOP_CLIP"
  exit 1
fi

if [[ ! -d "$CLIPS_DIR" ]]; then
  echo "Folder does not exist: $CLIPS_DIR"
  exit 1
fi

if command -v realpath >/dev/null 2>&1; then
  CLIPS_DIR="$(realpath "$CLIPS_DIR")"
fi

declare -a RECORDS=()
declare -A CHAIN_COUNT=()
declare -A CHAIN_MIN=()
declare -A CHAIN_MAX=()
declare -A CHAIN_EXAMPLE=()
declare -A CHAIN_IN_RANGE=()

while IFS= read -r -d '' path; do
  base="$(basename "$path")"
  # Capture: prefix + clip number + suffix (suffix starts with non-digit and includes extension)
  if [[ "$base" =~ ^(.*[^0-9]|)([0-9]+)([^0-9].*)$ ]]; then
    prefix="${BASH_REMATCH[1]}"
    clip_num_raw="${BASH_REMATCH[2]}"
    suffix="${BASH_REMATCH[3]}"
    clip_num=$((10#$clip_num_raw))
    key="${prefix}#${suffix}"
    RECORDS+=("${key}"$'\t'"${clip_num}"$'\t'"${path}")
    CHAIN_COUNT["$key"]=$(( ${CHAIN_COUNT["$key"]:-0} + 1 ))
    if [[ -z "${CHAIN_MIN["$key"]+x}" || $clip_num -lt ${CHAIN_MIN["$key"]} ]]; then
      CHAIN_MIN["$key"]=$clip_num
    fi
    if [[ -z "${CHAIN_MAX["$key"]+x}" || $clip_num -gt ${CHAIN_MAX["$key"]} ]]; then
      CHAIN_MAX["$key"]=$clip_num
    fi
    if [[ -z "${CHAIN_EXAMPLE["$key"]+x}" ]]; then
      CHAIN_EXAMPLE["$key"]="$base"
    fi
    if (( clip_num >= START_CLIP && clip_num <= STOP_CLIP )); then
      CHAIN_IN_RANGE["$key"]=$(( ${CHAIN_IN_RANGE["$key"]:-0} + 1 ))
    fi
  fi
done < <(
  find "$CLIPS_DIR" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.webm' \) \
    -print0
)

if (( ${#CHAIN_COUNT[@]} == 0 )); then
  echo "No clip chains detected in folder: $CLIPS_DIR"
  echo "Expected names with a numeric clip segment, e.g. Scene-0001_stereopilot_sbs.mp4"
  exit 1
fi

echo
echo "Detected clip chains in: $CLIPS_DIR"
echo "Clip range: $START_CLIP .. $STOP_CLIP"
echo

declare -a CHAIN_KEYS=()
i=1
while IFS= read -r key; do
  [[ -n "$key" ]] || continue
  CHAIN_KEYS+=("$key")
  prefix="${key%%#*}"
  suffix="${key#*#}"
  pattern="${prefix}<clip>${suffix}"
  total_count="${CHAIN_COUNT["$key"]}"
  in_range_count="${CHAIN_IN_RANGE["$key"]:-0}"
  min_clip="${CHAIN_MIN["$key"]}"
  max_clip="${CHAIN_MAX["$key"]}"
  example="${CHAIN_EXAMPLE["$key"]}"
  printf "  %2d) %s\n" "$i" "$pattern"
  printf "      clips=%s..%s  total=%s  usable_in_range=%s  example=%s\n" \
    "$min_clip" "$max_clip" "$total_count" "$in_range_count" "$example"
  i=$((i + 1))
done < <(printf '%s\n' "${!CHAIN_COUNT[@]}" | sort)

echo
read -r -p "Select chain number to combine: " CHOICE
if [[ ! "$CHOICE" =~ ^[0-9]+$ ]]; then
  echo "Invalid selection: $CHOICE"
  exit 1
fi
CHOICE_IDX=$((10#$CHOICE))
if (( CHOICE_IDX < 1 || CHOICE_IDX > ${#CHAIN_KEYS[@]} )); then
  echo "Selection out of range: $CHOICE"
  exit 1
fi

SELECTED_KEY="${CHAIN_KEYS[$((CHOICE_IDX - 1))]}"

declare -a SELECTED=()
for rec in "${RECORDS[@]}"; do
  rec_key="${rec%%$'\t'*}"
  rest="${rec#*$'\t'}"
  rec_num="${rest%%$'\t'*}"
  rec_path="${rest#*$'\t'}"
  if [[ "$rec_key" == "$SELECTED_KEY" ]] && (( rec_num >= START_CLIP && rec_num <= STOP_CLIP )); then
    SELECTED+=("${rec_num}"$'\t'"${rec_path}")
  fi
done

if (( ${#SELECTED[@]} == 0 )); then
  echo "No clips in selected chain within range $START_CLIP..$STOP_CLIP."
  exit 1
fi

mapfile -t SELECTED_SORTED < <(printf '%s\n' "${SELECTED[@]}" | sort -t $'\t' -k1,1n -k2,2)

first_num="${SELECTED_SORTED[0]%%$'\t'*}"
last_entry="${SELECTED_SORTED[$(( ${#SELECTED_SORTED[@]} - 1 ))]}"
last_num="${last_entry%%$'\t'*}"

suffix="${SELECTED_KEY#*#}"
ext="${suffix##*.}"
if [[ -z "$ext" || "$ext" == "$suffix" ]]; then
  ext="mp4"
fi

set_tag_raw="${SELECTED_KEY%%#*}${suffix%.*}"
set_tag="$(printf '%s' "$set_tag_raw" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//')"
if [[ -z "$set_tag" ]]; then
  set_tag="clips"
fi

OUTPUT_FILE="${CLIPS_DIR}/${set_tag}_from_${START_CLIP}_to_${STOP_CLIP}_combined.${ext}"
LIST_FILE="$(mktemp "${CLIPS_DIR}/concat_list_XXXXXX.txt")"
trap 'rm -f "$LIST_FILE"' EXIT

for row in "${SELECTED_SORTED[@]}"; do
  clip_path="${row#*$'\t'}"
  esc=${clip_path//\'/\'\\\'\'}
  printf "file '%s'\n" "$esc" >> "$LIST_FILE"
done

echo
echo "Combining ${#SELECTED_SORTED[@]} clips (clip ${first_num} .. ${last_num})"
echo "Output: $OUTPUT_FILE"
echo

ffmpeg -hide_banner -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT_FILE"

echo
echo "Done: $OUTPUT_FILE"
