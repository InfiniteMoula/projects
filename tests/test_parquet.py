
import pytest

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from utils.parquet import ParquetBatchWriter, iter_batches


def test_parquet_writer_and_iter(tmp_path):
    df = pd.DataFrame({"a": pd.Series(["1", "2"], dtype="string"), "b": pd.Series([1, 2], dtype="int64")})
    path = tmp_path / "data.parquet"
    with ParquetBatchWriter(path) as writer:
        writer.write_pandas(df, preserve_index=False)
    batches = list(iter_batches(path))
    assert len(batches) == 1
    assert batches[0]["a"].tolist() == ["1", "2"]
