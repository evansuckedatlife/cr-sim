"""Parser for Supercell's ``csv_logic`` dialect.

The format is a normal CSV with two peculiarities:

1. Row 0 is the column names, row 1 is the column *types*
   (``string`` / ``int`` / ``boolean``, sometimes capitalised inconsistently).
2. A row whose first column is non-empty starts a new **record**.  Rows whose
   first column is empty are **continuation rows** belonging to the record above
   them, and supply per-level (or per-index) values for the columns they fill.

So ``rarities.csv`` stores ``PowerLevelMultiplier`` as one value on the ``Common``
row plus one value on each of the continuation rows beneath it -- an array laid
out vertically.  Every column is therefore modelled as a tuple; scalar columns
are just tuples of length 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

__all__ = ["Record", "Table", "load_table", "CsvFormatError"]


class CsvFormatError(ValueError):
    """Raised when a file does not look like a Supercell csv_logic table."""


def _coerce(raw: str, typ: str) -> Any:
    """Convert one cell to its declared type. Empty cells are always ``None``.

    Array types (``IntArray``, ``StringArray``, ``BooleanArray``) hold one
    *element* per row -- the array is built vertically out of continuation rows
    -- so each cell is coerced to the element type and :meth:`Record.array`
    assembles it.
    """
    text = raw.strip()
    if not text:
        return None
    kind = typ.strip().lower().removesuffix("array")
    if kind == "int":
        try:
            return int(text)
        except ValueError:
            # A handful of columns are declared int but hold a float or a name.
            try:
                return int(float(text))
            except ValueError:
                return None
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            return None
    if kind == "boolean":
        return text.lower() == "true"
    return text


@dataclass(frozen=True, slots=True)
class Record:
    """One logical row: a name plus a column -> value-tuple mapping."""

    name: str
    index: int
    row_count: int
    columns: Mapping[str, tuple[Any, ...]]

    def scalar(self, column: str, default: Any = None) -> Any:
        """First (level-0) value of ``column``."""
        values = self.columns.get(column)
        if not values or values[0] is None:
            return default
        return values[0]

    def array(self, column: str) -> tuple[Any, ...]:
        """All values of ``column``, with trailing ``None``s trimmed."""
        values = self.columns.get(column, ())
        end = len(values)
        while end and values[end - 1] is None:
            end -= 1
        return values[:end]

    def at(self, column: str, level: int, default: Any = None) -> Any:
        """Value of ``column`` at continuation index ``level``, else ``default``."""
        values = self.columns.get(column, ())
        if level < 0 or level >= len(values) or values[level] is None:
            return default
        return values[level]

    def __contains__(self, column: str) -> bool:
        return self.columns.get(column, (None,))[0] is not None


@dataclass(frozen=True, slots=True)
class Table:
    """A parsed csv_logic file."""

    path: Path
    column_names: tuple[str, ...]
    column_types: tuple[str, ...]
    records: tuple[Record, ...]
    by_name: Mapping[str, Record]

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, name: str) -> Record:
        return self.by_name[name]

    def get(self, name: str, default: Record | None = None) -> Record | None:
        return self.by_name.get(name, default)

    def has_column(self, column: str) -> bool:
        return column in self.column_names


def _rows_from_text(text: str) -> list[list[str]]:
    return list(csv.reader(text.splitlines()))


def load_table(path: str | Path, *, text: str | None = None) -> Table:
    """Parse a csv_logic file into a :class:`Table`.

    ``text`` lets callers feed already-decoded bytes (see :mod:`cr_sim.data.decode`)
    without a second trip to disk.
    """
    path = Path(path)
    if text is None:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = _rows_from_text(text)
    if len(rows) < 2:
        raise CsvFormatError(f"{path.name}: need a header row and a type row")

    names = tuple(c.strip() for c in rows[0])
    types = tuple(rows[1])
    if len(types) < len(names):
        types = types + ("string",) * (len(names) - len(types))
    if not names or not names[0]:
        raise CsvFormatError(f"{path.name}: first column of the header row is empty")

    records: list[Record] = []
    # Per-record accumulator: column name -> list of values, one per row.
    pending_name: str | None = None
    pending_index = 0
    pending: dict[str, list[Any]] = {}

    def flush() -> None:
        nonlocal pending_name, pending
        if pending_name is None:
            return
        depth = max((len(v) for v in pending.values()), default=1)
        columns = {k: tuple(v) for k, v in pending.items()}
        records.append(
            Record(name=pending_name, index=pending_index, row_count=depth, columns=columns)
        )
        pending_name = None
        pending = {}

    for row in rows[2:]:
        if not row:
            continue
        first = row[0].strip() if row else ""
        if first:
            flush()
            pending_name = first
            pending_index = len(records)
            pending = {name: [] for name in names}
        elif pending_name is None:
            continue  # continuation row with no owner; skip
        for col_i, name in enumerate(names):
            raw = row[col_i] if col_i < len(row) else ""
            pending[name].append(_coerce(raw, types[col_i]))

    flush()

    by_name: dict[str, Record] = {}
    for record in records:
        by_name.setdefault(record.name, record)

    return Table(
        path=path,
        column_names=names,
        column_types=types,
        records=tuple(records),
        by_name=by_name,
    )
