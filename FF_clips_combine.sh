#!/usr/bin/env bash
set -euo pipefail

# ---- 1) Check ffmpeg is installed ----
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Please install it and ensure it is in your PATH."
  echo "Install: sudo apt update && sudo apt install -y ffmpeg"
  exit 1
fi

# ---- 2) Check an input file was provided ----
if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: $0 /path/to/any/video.mp4"
  echo "Tip: you can drag-and-drop a file into the terminal after typing '$0 '"
  exit 1
fi

INPUT_FILE="$1"

# ---- 3) Validate input exists ----
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file does not exist: $INPUT_FILE"
  exit 1
fi

# Resolve absolute path (best effort)
if command -v realpath >/dev/null 2>&1; then
  INPUT_FILE="$(realpath "$INPUT_FILE")"
fi

INPUT_DIR="$(dirname "$INPUT_FILE")"
INPUT_BASE="$(basename "$INPUT_FILE")"
INPUT_NAME="${INPUT_BASE%.*}"
OUTPUT_FILE="${INPUT_DIR}/${INPUT_NAME}_combined.mp4"
FILE_LIST="${INPUT_DIR}/file_list.txt"

# ---- 4) Build concat file list of all mp4s in the same directory ----
rm -f "$FILE_LIST"

# Sort deterministically (lexicographic). Matches typical Windows "for %%f" behavior.
# If you want natural sort (e.g. 1,2,10), replace sort with sort -V.
find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -print0 \
  | sort -z \
  | while IFS= read -r -d '' f; do
      # ffmpeg concat demuxer expects: file 'path'
      # Escape single quotes for safety
      esc=${f//\'/\'\\\'\'}
      printf "file '%s'\n" "$esc" >> "$FILE_LIST"
    done

if [[ ! -s "$FILE_LIST" ]]; then
  echo "No .mp4 files found in directory: $INPUT_DIR"
  exit 1
fi

# ---- 5) Combine videos using FFmpeg (stream copy) ----
ffmpeg -hide_banner -f concat -safe 0 -i "$FILE_LIST" -c copy "$OUTPUT_FILE"

# ---- 6) Cleanup ----
rm -f "$FILE_LIST"

echo "-------------------------"
echo
echo "Combined video created: $OUTPUT_FILE"

# Similar to `timeout /t 5`
sleep 5

