"""Select the OpTC eCAR files of download option A from a Drive listing (id, bytes, path)."""
import json
import sys

KEEP = ("ecar/evaluation/23Sep19-red/", "ecar/evaluation/24Sep19/", "ecar/evaluation/25Sept/", "ecar/benign/19Sep19/")
d = json.load(open(sys.argv[1]))
sel = [x for x in d if x["path"].startswith(KEEP)]
with open(sys.argv[2], "w", newline="\n") as f:
    for x in sel:
        f.write(f"{x['id']}\t{x['bytes']}\t{x['path']}\n")
print(len(sel), "files", round(sum(x["bytes"] for x in sel) / 1e9, 1), "GB; largest", max(x["bytes"] for x in sel) / 1e9)
