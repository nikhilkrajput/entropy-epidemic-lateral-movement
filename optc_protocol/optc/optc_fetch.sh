#!/bin/bash
# Download, filter and delete the OpTC eCAR files listed in $LIST (default optc_files_optionA.tsv)
# (id <TAB> bytes <TAB> path), PAR files at a time.  Resumable: a file whose
# flows/<name>.done marker exists is skipped; a download whose size differs
# from the listing is discarded and logged as failed (Drive quota errors are
# retried on the next run).
# Usage: ./optc_fetch.sh [PAR]
cd "$(dirname "$0")"
export PYTHONPATH=$HOME/optc/pylib:$HOME/lanl_cyber1/pylib
PAR=${1:-4}
one() {
  id=$1; bytes=$2; path=$3
  name=$(echo "$path" | sed 's#^ecar/##; s#/#__#g; s#\.json\.gz$##')
  [ -f "flows/$name.done" ] && return 0
  raw="raw/$name.json.gz"
  # Drive's direct endpoint; -C - resumes a partial file.  A quota-blocked file
  # returns a small HTML page, which fails the size check below.
  if [ "${FETCH:-curl}" = rclone ]; then
    # authenticated read-only Drive access (remote "optc", see README_optc.md)
    $HOME/optc/bin/rclone backend copyid optc: "$id" "$raw" --retries 5 >> "logs/$name.log" 2>&1
  else
    curl -sL --retry 5 --retry-delay 30 -C - -o "$raw" \
      "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" >> "logs/$name.log" 2>&1
  fi
  got=$(stat -c %s "$raw" 2>/dev/null || echo 0)
  if [ "$got" -lt 100000 ] && [ "$got" != "$bytes" ]; then rm -f "$raw"; fi
  if [ "$got" != "$bytes" ]; then
    echo "$(date -u +%FT%TZ) FAIL size $got != $bytes $path" >> logs/fetch.log; rm -f "$raw"; return 1
  fi
  if pigz -dc -p 2 "$raw" | python3 optc_filter.py labels/optc_redteam.csv "flows/$name.parquet" 2>> "logs/$name.log"; then
    touch "flows/$name.done"; echo "$(date -u +%FT%TZ) OK $path $(tail -1 logs/$name.log)" >> logs/fetch.log
  else
    echo "$(date -u +%FT%TZ) FAIL filter $path" >> logs/fetch.log
  fi
  rm -f "$raw"
}
export -f one
mkdir -p raw flows logs
tr '\t' ' ' < ${LIST:-optc_files_optionA.tsv} | xargs -P "$PAR" -L 1 bash -c 'one "$0" "$1" "$2"'
echo "$(date -u +%FT%TZ) pass done: $(ls flows/*.done 2>/dev/null | wc -l) of $(wc -l < ${LIST:-optc_files_optionA.tsv})" >> logs/fetch.log
