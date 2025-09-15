
"""Shared parquet reading/writing helpers."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

LOGGER = logging.getLogger("utils.parquet")


def _filter_columns(dataset: ds.Dataset, columns: Optional[Sequence[str]]) -> Optional[list[str]]:
    if columns is None:
        return None
    available = set(dataset.schema.names)
    selected = [col for col in columns if col in available]
    return selected or None


def iter_batches(
    path: Path | str,
    *,
    columns: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Yield pandas DataFrames for each record batch in a parquet dataset."""

    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(
        columns=_filter_columns(dataset, columns),
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        yield batch.to_pandas(types_mapper=pd.ArrowDtype)


class ParquetBatchWriter:
    """Lazy parquet writer that keeps schema aligned across batches."""

    def __init__(
        self,
        path: Path | str,
        *,
        schema: Optional[pa.Schema] = None,
        compression: str = "snappy",
    ) -> None:
        self._path = Path(path)
        self._schema = schema
        self._compression = compression
        self._writer: Optional[pq.ParquetWriter] = None

    def _ensure_writer(self, table: pa.Table) -> pq.ParquetWriter:
        if self._writer is None:
            schema = self._schema or table.schema
            self._schema = schema
            self._writer = pq.ParquetWriter(
                str(self._path),
                schema,
                compression=self._compression,
            )
        return self._writer

    def write_table(self, table: pa.Table) -> None:
        writer = self._ensure_writer(table)
        if self._schema is not None and table.schema != self._schema:
            table = table.cast(self._schema, safe=False)
        writer.write_table(table)

    def write_pandas(self, frame: pd.DataFrame, *, preserve_index: bool = False) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=preserve_index)
        self.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ArrowCsvWriter:
    """Append-friendly CSV writer backed by pyarrow."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._has_written = False
        self._schema: Optional[pa.Schema] = None

    def write_table(self, table: pa.Table) -> None:
        if self._schema is None:
            self._schema = table.schema
        mode = "wb" if not self._has_written else "ab"
        write_options = pa_csv.WriteOptions(include_header=not self._has_written)
        with self._path.open(mode) as sink:
            pa_csv.write_csv(table.cast(self._schema, safe=False), sink, write_options=write_options)
        self._has_written = True

    def close(self) -> None:  # parity with context manager usage
        self._schema = self._schema  # no-op

    def __enter__(self) -> "ArrowCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def parquet_writer(
    path: Path | str,
    *,
    schema: Optional[pa.Schema] = None,
    compression: str = "snappy",
) -> Iterator[ParquetBatchWriter]:
    writer = ParquetBatchWriter(path, schema=schema, compression=compression)
    try:
        yield writer
    finally:
        writer.close()
