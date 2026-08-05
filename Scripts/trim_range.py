#!/usr/bin/env python3
"""Trim a range map for the website.

`godwit range` writes a map meant for the dashboard, which is served from the
same machine that produced it and can afford to be verbose. The GitHub Pages
site ships the same data over the network to a first-time visitor, where half a
megabyte of JSON is the largest thing on the page.

This shortens the field names, drops what the site does not draw, replaces the
topic string on every point with an index into the topic list, and rounds the
coordinates to four decimal places — well beyond what a few hundred pixels of
canvas can resolve.

    python3 Scripts/trim_range.py model.gwt/range.json site/range.json

It existed only as a shell one-liner before, which meant the published map
could not be regenerated from a rebuilt one without reconstructing the command
from memory.
"""

import json
import pathlib
import sys


def trim(source: dict) -> dict:
    topics = source["topics"]
    index_of = {name: i for i, name in enumerate(topics)}

    points = []
    for p in source["points"]:
        point = {
            "l": p["layer"],
            "e": p["expert"],
            "x": round(p["x"], 4),
            "y": round(p["y"], 4),
            "z": round(p["z"], 4),
            "t": index_of[p["topic"]],
            "s": round(p["specialisation"], 4),
            "a": p["activations"],
        }
        # Only carried when false, since that is the rarer case and the reader
        # treats a missing flag as confident.
        if not p.get("confident", True):
            point["c"] = 0
        points.append(point)

    out = {
        "topics": topics,
        "variance": [round(v, 4) for v in source["variance"]],
        "points": points,
    }
    if "minimumActivations" in source:
        out["min"] = source["minimumActivations"]
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {sys.argv[0]} <range.json> <output.json>", file=sys.stderr)
        return 2

    source = pathlib.Path(sys.argv[1])
    target = pathlib.Path(sys.argv[2])

    data = json.loads(source.read_text())
    trimmed = trim(data)
    target.write_text(json.dumps(trimmed, separators=(",", ":")))

    before, after = source.stat().st_size, target.stat().st_size
    unsure = sum(1 for p in trimmed["points"] if p.get("c") == 0)
    print(f"{len(trimmed['points'])} experts, {unsure} below the activation bar")
    print(f"{before / 1024:.0f} KiB -> {after / 1024:.0f} KiB "
          f"({100 * after / before:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
