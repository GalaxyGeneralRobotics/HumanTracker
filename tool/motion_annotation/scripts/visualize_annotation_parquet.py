#!/usr/bin/env python3
"""Render one canonical annotation Parquet shard as a standalone HTML table."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def read_records(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["annotation_json"])
    records = [json.loads(value) for value in table["annotation_json"].to_pylist()]
    return records


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    candidates = record["candidates"]
    if [int(item["candidate_idx"]) for item in candidates] != [0, 1]:
        raise ValueError(f"{record['record_id']}: candidate_idx must be [0, 1]")
    order = [int(value) for value in record["comparison"]["display_order"]]
    if order not in ([0, 1], [1, 0]):
        raise ValueError(f"{record['record_id']}: invalid display_order")
    choice_type = str(record["preference"]["choice_type"])
    preferred = record["preference"]["preferred_candidate_idx"]
    if choice_type == "preference":
        preferred = int(preferred)
        canonical_choice = f"candidate_{preferred}"
        display_choice = "left" if order[0] == preferred else "right"
    else:
        canonical_choice = choice_type
        display_choice = choice_type
    by_idx = {int(candidate["candidate_idx"]): candidate for candidate in candidates}
    extra = candidates[0]["extra"]
    return {
        "record_id": str(record["record_id"]),
        "timestamp": str(record["meta"]["timestamp"]),
        "annotator_id": str(record["meta"]["annotator_id"]),
        "pair_id": str(record["pair_id"]),
        "motion_id": str(record["motion_id"]),
        "category": str(record["category"]),
        "clip_uid": str(extra["clip_uid"]),
        "display_choice": display_choice,
        "canonical_choice": canonical_choice,
        "left_tracker": str(by_idx[order[0]]["tracker"]),
        "right_tracker": str(by_idx[order[1]]["tracker"]),
        "display_order": order,
        "source_start_frame": int(extra["source_start_frame"]),
        "source_end_frame": int(extra["source_end_frame"]),
        "invalid": bool(record["flags"]["invalid"]),
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render(path: Path, rows: list[dict[str, Any]]) -> str:
    choices = Counter(row["canonical_choice"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    reversed_count = sum(row["display_order"] == [1, 0] for row in rows)
    cards = [
        ("Records", len(rows)),
        ("Preference", choices["candidate_0"] + choices["candidate_1"]),
        ("Similar", choices["similar"]),
        ("Bad traj", choices["bad_traj"]),
        ("Reversed display", reversed_count),
        ("Categories", len(categories)),
    ]
    card_html = "".join(
        f'<div class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in cards
    )
    body_rows = "".join(
        "<tr>"
        f'<td>{index}</td><td>{esc(row["canonical_choice"])}</td>'
        f'<td>{esc(row["display_choice"])}</td>'
        f'<td>{esc(row["left_tracker"])}</td><td>{esc(row["right_tracker"])}</td>'
        f'<td>{esc(row["display_order"])}</td><td>{esc(row["category"])}</td>'
        f'<td class="mono">{esc(row["motion_id"])}</td>'
        f'<td class="mono">{esc(row["pair_id"])}</td>'
        f'<td>{esc(row["annotator_id"])}</td><td>{esc(row["timestamp"])}</td>'
        "</tr>"
        for index, row in enumerate(rows)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation shard: {esc(path.name)}</title>
<style>
body{{font:14px system-ui;margin:0;background:#f5f6f8;color:#172033}}main{{padding:24px}}
h1{{margin:0 0 4px}}.path{{color:#657086;margin-bottom:20px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}
.card{{background:white;border:1px solid #dfe3ea;border-radius:8px;padding:10px 14px;min-width:130px}}
.card span{{display:block;color:#657086;font-size:12px}}.card strong{{font-size:22px}}
input{{width:min(620px,100%);padding:10px;margin:18px 0;border:1px solid #cbd2dc;border-radius:7px}}
.table{{overflow:auto;background:white;border:1px solid #dfe3ea;border-radius:8px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px 10px;border-bottom:1px solid #edf0f4;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#f8f9fb}}
.mono{{font-family:ui-monospace,monospace;font-size:12px}}
</style></head><body><main><h1>Canonical annotation shard</h1><div class="path">{esc(path)}</div>
<div class="cards">{card_html}</div><input id="search" placeholder="Filter rows">
<div class="table"><table><thead><tr><th>#</th><th>Canonical</th><th>Display</th><th>Left tracker</th><th>Right tracker</th><th>Order</th><th>Category</th><th>Motion</th><th>Pair</th><th>Annotator</th><th>Time</th></tr></thead>
<tbody id="rows">{body_rows}</tbody></table></div></main>
<script>document.getElementById('search').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();for(const row of document.querySelectorAll('#rows tr'))row.hidden=!row.innerText.toLowerCase().includes(q);}});</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.parquet.resolve()
    output = args.output.resolve() if args.output else source.with_suffix(".html")
    output.write_text(
        render(source, [summarize(record) for record in read_records(source)]),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
