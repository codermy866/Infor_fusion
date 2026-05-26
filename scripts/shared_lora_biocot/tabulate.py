"""Small local fallback for pandas.DataFrame.to_markdown().

The formal LOCO runner only needs pipe-style Markdown tables for audit reports.
Some locked experiment environments do not allow installing the third-party
``tabulate`` package, so this shim implements the tiny subset used by pandas.
"""

from __future__ import annotations

__version__ = "0.10.0"


def _stringify(value) -> str:
    if value is None:
        return ""
    return str(value)


def tabulate(
    tabular_data,
    headers=(),
    tablefmt: str = "pipe",
    showindex="default",
    **_,
) -> str:
    if tablefmt not in {"pipe", "github"}:
        tablefmt = "pipe"

    if hasattr(tabular_data, "columns") and hasattr(tabular_data, "itertuples"):
        frame = tabular_data
        header_values = list(frame.columns) if headers in (None, "keys", ()) else list(headers)
        rows = [list(row) for row in frame.itertuples(index=False, name=None)]
    else:
        rows = []
        for row in tabular_data:
            if isinstance(row, dict):
                rows.append([row.get(h, "") for h in headers])
            else:
                rows.append(list(row))
        header_values = list(headers) if headers not in (None, "keys") else []
    if not header_values and rows:
        header_values = [str(i) for i in range(len(rows[0]))]

    if showindex not in (False, "never") and rows:
        header_values = [""] + header_values
        rows = [[i] + row for i, row in enumerate(rows)]

    str_rows = [[_stringify(v) for v in row] for row in rows]
    str_headers = [_stringify(v) for v in header_values]
    n_cols = max([len(str_headers)] + [len(row) for row in str_rows] + [0])
    str_headers += [""] * (n_cols - len(str_headers))
    str_rows = [row + [""] * (n_cols - len(row)) for row in str_rows]

    widths = [
        max(len(str_headers[col]), *(len(row[col]) for row in str_rows)) if str_rows else len(str_headers[col])
        for col in range(n_cols)
    ]

    def fmt(row):
        cells = [row[col].ljust(widths[col]) for col in range(n_cols)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * max(width, 3) for width in widths) + " |"
    lines = [fmt(str_headers), separator]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)
