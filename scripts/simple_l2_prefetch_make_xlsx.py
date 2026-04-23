#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


CASE_ORDER = [
    "vanilla_baseline",
    "experiment_off",
    "experiment_prefetch_a1_128k",
    "experiment_prefetch_a1_256k",
    "experiment_prefetch_a1_512k",
    "experiment_prefetch_a2_128k",
    "experiment_prefetch_a2_256k",
    "experiment_prefetch_a2_512k",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create an Excel summary from SIMPLE L2 prefetch logs.")
    parser.add_argument("result_dir", type=Path, help="Result directory containing *.log files")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .xlsx path")
    return parser.parse_args()


def case_config(name):
    if name == "vanilla_baseline":
        return "vanilla", 0, 0, 1
    if name == "experiment_off":
        return "experiment", 0, 0, 1
    match = re.match(r"experiment_prefetch_a([0-9]+)_([0-9]+)k$", name)
    if match:
        ahead = int(match.group(1))
        max_bytes = int(match.group(2)) * 1024
        return "experiment", 1, max_bytes, ahead
    return "unknown", "", "", ""


def parse_log(path):
    rows = {}
    avg_busbw = None
    nccl_line = ""
    devices = []
    row_re = re.compile(r"^\s*([0-9]+)\s+")
    with path.open("r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("# nccl-tests version"):
                nccl_line = stripped.lstrip("# ").strip()
            elif stripped.startswith("#  Rank"):
                devices.append(stripped.lstrip("# ").strip())
            elif stripped.startswith("# Avg bus bandwidth"):
                try:
                    avg_busbw = float(stripped.split(":")[-1])
                except ValueError:
                    avg_busbw = None
            elif row_re.match(line):
                fields = stripped.split()
                if len(fields) < 13:
                    continue
                size = int(fields[0])
                rows[size] = {
                    "count": int(fields[1]),
                    "type": fields[2],
                    "redop": fields[3],
                    "root": fields[4],
                    "oop_time_us": float(fields[5]),
                    "oop_algbw_gbps": float(fields[6]),
                    "oop_busbw_gbps": float(fields[7]),
                    "oop_wrong": int(fields[8]),
                    "ip_time_us": float(fields[9]),
                    "ip_algbw_gbps": float(fields[10]),
                    "ip_busbw_gbps": float(fields[11]),
                    "ip_wrong": int(fields[12]),
                }
    return {"rows": rows, "avg_busbw": avg_busbw, "nccl_line": nccl_line, "devices": devices}


def percent_delta(new, old):
    if new is None or old in (None, 0):
        return None
    return 100.0 * (new - old) / old


def collect_cases(result_dir):
    cases = {}
    for name in CASE_ORDER:
        path = result_dir / f"{name}.log"
        if path.exists():
            cases[name] = parse_log(path)
            cases[name]["log"] = path.name
            cases[name]["config"] = case_config(name)
    extra_logs = sorted(p for p in result_dir.glob("*.log") if p.stem not in cases)
    for path in extra_logs:
        cases[path.stem] = parse_log(path)
        cases[path.stem]["log"] = path.name
        cases[path.stem]["config"] = case_config(path.stem)
    return cases


def build_summary(cases):
    vanilla = cases.get("vanilla_baseline", {}).get("avg_busbw")
    exp_off = cases.get("experiment_off", {}).get("avg_busbw")
    rows = [["case", "repo", "prefetch", "ahead_chunks", "max_bytes", "avg_busbw_gbps", "vs_vanilla_pct", "vs_experiment_off_pct", "log"]]
    for name, data in cases.items():
        repo, enable, max_bytes, ahead = data["config"]
        avg = data.get("avg_busbw")
        rows.append([
            name,
            repo,
            enable,
            ahead,
            max_bytes,
            avg,
            percent_delta(avg, vanilla),
            percent_delta(avg, exp_off),
            data.get("log", ""),
        ])
    return rows


def all_sizes(cases):
    sizes = set()
    for data in cases.values():
        sizes.update(data["rows"].keys())
    return sorted(sizes)


def build_metric_sheet(cases, metric, title):
    sizes = all_sizes(cases)
    names = list(cases.keys())
    rows = [["size_bytes", "size_mib"] + names + ["best_case", "best_value"]]
    for size in sizes:
        values = [cases[name]["rows"].get(size, {}).get(metric) for name in names]
        best_name = ""
        best_value = None
        for name, value in zip(names, values):
            if value is None:
                continue
            if best_value is None:
                best_name = name
                best_value = value
            elif ("time" in metric and value < best_value) or ("time" not in metric and value > best_value):
                best_name = name
                best_value = value
        rows.append([size, size / (1024.0 * 1024.0)] + values + [best_name, best_value])
    return title, rows


def build_delta_sheet(cases):
    sizes = all_sizes(cases)
    names = list(cases.keys())
    vanilla = cases.get("vanilla_baseline", {})
    rows = [["size_bytes", "size_mib"] + [f"{name}_vs_vanilla_pct" for name in names if name != "vanilla_baseline"]]
    for size in sizes:
        base = vanilla.get("rows", {}).get(size, {}).get("oop_busbw_gbps")
        row = [size, size / (1024.0 * 1024.0)]
        for name in names:
            if name == "vanilla_baseline":
                continue
            value = cases[name]["rows"].get(size, {}).get("oop_busbw_gbps")
            row.append(percent_delta(value, base))
        rows.append(row)
    return rows


def build_metadata(result_dir, cases):
    rows = [
        ["field", "value"],
        ["generated_at", dt.datetime.now().isoformat(timespec="seconds")],
        ["result_dir", str(result_dir)],
        ["case_count", len(cases)],
    ]
    for name, data in cases.items():
        rows.append([f"{name}_nccl", data.get("nccl_line", "")])
    return rows


def col_name(index):
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row_idx, col_idx, value, style=0):
    ref = f"{col_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{style_attr} t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return f'<c r="{ref}"{style_attr}/>'
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{text}</t></is></c>'


def sheet_xml(rows, freeze=True, percent_cols=None):
    percent_cols = percent_cols or set()
    max_cols = max((len(r) for r in rows), default=1)
    widths = []
    for c in range(max_cols):
        max_len = 10
        for row in rows[:100]:
            if c < len(row) and row[c] is not None:
                max_len = max(max_len, len(str(row[c])) + 2)
        widths.append(min(max_len, 32))

    cols = "".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    sheet_views = ""
    if freeze:
        sheet_views = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    body = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            if r_idx == 1:
                style = 1
            elif c_idx in percent_cols:
                style = 3
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                style = 2
            else:
                style = 0
            cells.append(cell_xml(r_idx, c_idx, value, style))
        body.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    auto_filter = f'<autoFilter ref="A1:{col_name(max_cols)}{len(rows)}"/>' if rows else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'{sheet_views}{cols}<sheetData>{"".join(body)}</sheetData>{auto_filter}</worksheet>'
    )


def styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="10" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def workbook_xml(sheet_names):
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(sheets)}</sheets></workbook>'
    )


def workbook_rels_xml(sheet_count):
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>')
    rels.append(f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}</Relationships>'
    )


def root_rels_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''


def content_types_xml(sheet_count):
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{sheet_overrides}</Types>'
    )


def write_xlsx(path, sheets):
    sheet_names = [name for name, _rows, _pct_cols in sheets]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        z.writestr("_rels/.rels", root_rels_xml())
        z.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        z.writestr("xl/styles.xml", styles_xml())
        for idx, (_name, rows, pct_cols) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows, percent_cols=pct_cols))


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = args.output or (result_dir / "simple_l2_prefetch_summary.xlsx")
    cases = collect_cases(result_dir)
    if not cases:
        raise SystemExit(f"No log files found under {result_dir}")

    summary = build_summary(cases)
    sheets = [
        ("Summary", summary, {7, 8}),
        build_metric_sheet(cases, "oop_busbw_gbps", "OOP_BusBW") + (set(),),
        ("OOP_DeltaPct", build_delta_sheet(cases), set(range(3, 3 + max(len(cases) - 1, 0)))),
        build_metric_sheet(cases, "oop_time_us", "OOP_Time_us") + (set(),),
        build_metric_sheet(cases, "ip_busbw_gbps", "IP_BusBW") + (set(),),
        ("Metadata", build_metadata(result_dir, cases), set()),
    ]
    write_xlsx(output, sheets)
    print(output)


if __name__ == "__main__":
    main()
