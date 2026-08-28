"""
Compatibility layer for DataFrames and Numerical operations.
Seamlessly imports real `pandas` and `numpy` when installed;
provides robust standard-library fallbacks when running in minimal Python environments.
"""

import copy
import math
import numbers
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# =============================================================================
# Try importing real pandas and numpy
# =============================================================================

_HAS_PANDAS = False
_HAS_NUMPY = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None


# =============================================================================
# Fallback Numpy Shim
# =============================================================================

if not _HAS_NUMPY:

    class _FakeNumpy:
        nan = float("nan")
        number = (int, float, numbers.Number)

        @staticmethod
        def mean(a: Sequence[float]) -> float:
            lst = [float(x) for x in a if x is not None and not math.isnan(x)]
            return sum(lst) / len(lst) if lst else 0.0

        @staticmethod
        def median(a: Sequence[float]) -> float:
            lst = sorted([float(x) for x in a if x is not None and not math.isnan(x)])
            if not lst:
                return 0.0
            n = len(lst)
            mid = n // 2
            if n % 2 == 1:
                return lst[mid]
            return (lst[mid - 1] + lst[mid]) / 2.0

        @staticmethod
        def percentile(a: Sequence[float], q: float) -> float:
            lst = sorted([float(x) for x in a if x is not None and not math.isnan(x)])
            if not lst:
                return 0.0
            if q <= 0:
                return lst[0]
            if q >= 100:
                return lst[-1]
            k = (len(lst) - 1) * (q / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return lst[int(k)]
            d0 = lst[int(f)] * (c - k)
            d1 = lst[int(c)] * (k - f)
            return d0 + d1

        @staticmethod
        def min(a: Sequence[float]) -> float:
            lst = [float(x) for x in a if x is not None and not math.isnan(x)]
            return min(lst) if lst else 0.0

        @staticmethod
        def max(a: Sequence[float]) -> float:
            lst = [float(x) for x in a if x is not None and not math.isnan(x)]
            return max(lst) if lst else 0.0

        @staticmethod
        def sum(a: Sequence[float]) -> float:
            lst = [float(x) for x in a if x is not None and not math.isnan(x)]
            return sum(lst)

        @staticmethod
        def array(a: Sequence[Any], dtype=None):
            return list(a)

    np = _FakeNumpy()


# =============================================================================
# Fallback Pandas Shim
# =============================================================================

if not _HAS_PANDAS:

    class Series:
        """Lightweight standard library Series shim."""

        def __init__(self, data: Sequence[Any], name: Optional[str] = None):
            self._data = list(data)
            self.name = name

        def tolist(self) -> List[Any]:
            return list(self._data)

        def to_dict(self) -> Dict[int, Any]:
            return {i: v for i, v in enumerate(self._data)}

        def apply(self, func: Callable[[Any], Any]) -> "Series":
            return Series([func(x) for x in self._data], name=self.name)

        def round(self, decimals: int = 4) -> "Series":
            res = []
            for x in self._data:
                try:
                    res.append(round(float(x), decimals))
                except Exception:
                    res.append(x)
            return Series(res, name=self.name)

        def astype(self, dtype: Any) -> "Series":
            res = []
            for x in self._data:
                if dtype is str or dtype == "str" or dtype == "string":
                    res.append(str(x) if x is not None else "")
                elif dtype is bool or dtype == "bool":
                    res.append(bool(x))
                elif dtype is float or dtype == "float":
                    try:
                        res.append(float(x))
                    except Exception:
                        res.append(float("nan"))
                elif dtype is int or dtype == "int":
                    try:
                        res.append(int(x))
                    except Exception:
                        res.append(0)
                else:
                    res.append(x)
            return Series(res, name=self.name)

        @property
        def str(self):
            class _StrAccessor:
                def __init__(self, series_data):
                    self._data = series_data

                def strip(self):
                    return Series([str(x).strip() if x is not None else "" for x in self._data])

                def lower(self):
                    return Series([str(x).lower() if x is not None else "" for x in self._data])

            return _StrAccessor(self._data)

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> Any:
            return self._data[idx]

        def __iter__(self):
            return iter(self._data)

        def __repr__(self) -> str:
            return f"Series({self._data})"

    class DataFrame:
        """Lightweight standard library DataFrame shim."""

        def __init__(
            self,
            data: Any = None,
            columns: Optional[Sequence[str]] = None,
        ):
            self._records: List[Dict[str, Any]] = []
            self._columns: List[str] = list(columns) if columns is not None else []

            if data is None:
                pass
            elif isinstance(data, DataFrame):
                self._records = copy.deepcopy(data._records)
                self._columns = list(data._columns)
            elif isinstance(data, list):
                if data and isinstance(data[0], dict):
                    self._records = [dict(r) for r in data]
                    if not self._columns:
                        self._columns = list(data[0].keys())
                elif data and isinstance(data[0], (list, tuple)):
                    if not self._columns:
                        self._columns = [f"col_{i}" for i in range(len(data[0]))]
                    self._records = [dict(zip(self._columns, row)) for row in data]
                else:
                    # List of scalars
                    if not self._columns:
                        self._columns = ["0"]
                    self._records = [{self._columns[0]: val} for val in data]
            elif isinstance(data, dict):
                # Dict of lists or dict of scalar
                keys = list(data.keys())
                if not self._columns:
                    self._columns = keys
                if keys and isinstance(data[keys[0]], (list, tuple, Series)):
                    num_rows = len(data[keys[0]])
                    self._records = [
                        {k: (data[k][i] if i < len(data[k]) else None) for k in keys}
                        for i in range(num_rows)
                    ]
                else:
                    self._records = [dict(data)]

            # Ensure all records have all columns
            for r in self._records:
                for col in self._columns:
                    if col not in r:
                        r[col] = None

        @property
        def columns(self) -> List[str]:
            return self._columns

        @columns.setter
        def columns(self, new_cols: Sequence[str]):
            new_list = list(new_cols)
            if len(new_list) == len(self._columns):
                old_to_new = dict(zip(self._columns, new_list))
                new_records = []
                for r in self._records:
                    new_r = {old_to_new.get(k, k): v for k, v in r.items()}
                    new_records.append(new_r)
                self._records = new_records
            self._columns = new_list

        @property
        def shape(self) -> Tuple[int, int]:
            return len(self._records), len(self._columns)

        @property
        def empty(self) -> bool:
            return len(self._records) == 0

        def copy(self) -> "DataFrame":
            return DataFrame(data=[dict(r) for r in self._records], columns=list(self._columns))

        def reset_index(self, drop: bool = True) -> "DataFrame":
            return self.copy()

        def reindex(self, columns: Optional[Sequence[str]] = None, axis: int = 1) -> "DataFrame":
            new_cols = list(columns) if columns is not None else list(self._columns)
            new_recs = []
            for r in self._records:
                new_r = {c: r.get(c) for c in new_cols}
                new_recs.append(new_r)
            return DataFrame(new_recs, columns=new_cols)

        def sort_values(self, by: Union[str, Sequence[str]], ascending: bool = True) -> "DataFrame":
            by_cols = [by] if isinstance(by, str) else list(by)

            def sort_key(rec):
                key = []
                for c in by_cols:
                    val = rec.get(c)
                    if val is None:
                        key.append((0, ""))
                    elif isinstance(val, (int, float)):
                        key.append((1, float(val)))
                    else:
                        key.append((2, str(val)))
                return tuple(key)

            sorted_recs = sorted(self._records, key=sort_key, reverse=not ascending)
            return DataFrame(sorted_recs, columns=list(self._columns))

        def select_dtypes(self, include: Optional[Sequence[Any]] = None) -> "DataFrame":
            inc = include or []
            selected_cols = []
            for col in self._columns:
                vals = [r.get(col) for r in self._records if r.get(col) is not None]
                if "number" in inc or np.number in inc:
                    if all(isinstance(v, (int, float, numbers.Number)) for v in vals):
                        selected_cols.append(col)
                elif "object" in inc or "string" in inc or str in inc:
                    if any(isinstance(v, str) for v in vals):
                        selected_cols.append(col)
            return DataFrame(self._records, columns=selected_cols)

        def astype(self, dtype: Any) -> "DataFrame":
            new_recs = []
            for r in self._records:
                new_r = {}
                for c, v in r.items():
                    if dtype is str or dtype == "str":
                        new_r[c] = str(v) if v is not None else ""
                    else:
                        new_r[c] = v
                new_recs.append(new_r)
            return DataFrame(new_recs, columns=list(self._columns))

        def to_dict(self, orient: str = "records") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
            if orient == "records":
                return [dict(r) for r in self._records]
            elif orient == "list":
                return {c: [r.get(c) for r in self._records] for c in self._columns}
            return [dict(r) for r in self._records]

        def equals(self, other: Any) -> bool:
            if not isinstance(other, DataFrame):
                return False
            if self.shape != other.shape or list(self._columns) != list(other._columns):
                return False
            return self._records == other._records

        def __getitem__(self, item: Union[str, Sequence[str]]) -> Any:
            if isinstance(item, (list, tuple)):
                sub_cols = list(item)
                sub_recs = [{c: r.get(c) for c in sub_cols} for r in self._records]
                return DataFrame(sub_recs, columns=sub_cols)
            vals = [r.get(item) for r in self._records]
            return Series(vals, name=str(item))

        def __setitem__(self, key: str, value: Any):
            if key not in self._columns:
                self._columns.append(key)
            if isinstance(value, Series):
                for i, r in enumerate(self._records):
                    r[key] = value._data[i] if i < len(value._data) else None
            elif isinstance(value, (list, tuple)):
                for i, r in enumerate(self._records):
                    r[key] = value[i] if i < len(value) else None
            else:
                for r in self._records:
                    r[key] = value

        @property
        def loc(self):
            class _LocAccessor:
                def __init__(self, df):
                    self._df = df

                def __getitem__(self, idx_list):
                    if isinstance(idx_list, (list, tuple)):
                        recs = [
                            self._df._records[i] for i in idx_list if i < len(self._df._records)
                        ]
                        return DataFrame(recs, columns=self._df._columns)
                    elif isinstance(idx_list, int):
                        return self._df._records[idx_list]
                    return self._df

            return _LocAccessor(self)

        def __len__(self) -> int:
            return len(self._records)

        def __repr__(self) -> str:
            return f"DataFrame(columns={self._columns}, rows={len(self._records)})"

    class _ApiTypes:
        @staticmethod
        def is_numeric_dtype(series: Any) -> bool:
            vals = [
                x for x in (series._data if hasattr(series, "_data") else series) if x is not None
            ]
            if not vals:
                return False
            return all(
                isinstance(x, (int, float, numbers.Number)) and not isinstance(x, bool)
                for x in vals
            )

        @staticmethod
        def is_bool_dtype(series: Any) -> bool:
            vals = [
                x for x in (series._data if hasattr(series, "_data") else series) if x is not None
            ]
            if not vals:
                return False
            return all(isinstance(x, bool) for x in vals)

        @staticmethod
        def is_datetime64_any_dtype(series: Any) -> bool:
            return False

    class _PandasApi:
        types = _ApiTypes()

    class _Testing:
        @staticmethod
        def assert_frame_equal(
            df1: Any,
            df2: Any,
            check_like: bool = True,
            atol: float = 1e-4,
            rtol: float = 1e-4,
            check_dtype: bool = False,
            check_exact: bool = False,
            **kwargs,
        ):
            """Pure Python implementation of assert_frame_equal for fallback."""
            if df1.shape != df2.shape:
                raise AssertionError(f"DataFrame shapes are different: {df1.shape} vs {df2.shape}")

            cols1 = list(df1.columns)
            cols2 = list(df2.columns)

            if check_like:
                if sorted(cols1) != sorted(cols2):
                    raise AssertionError(
                        f"DataFrame column names are different: {cols1} vs {cols2}"
                    )
                cols = sorted(cols1)
            else:
                if cols1 != cols2:
                    raise AssertionError(f"DataFrame column order is different: {cols1} vs {cols2}")
                cols = cols1

            recs1 = df1.to_dict(orient="records") if hasattr(df1, "to_dict") else df1
            recs2 = df2.to_dict(orient="records") if hasattr(df2, "to_dict") else df2

            for i, (r1, r2) in enumerate(zip(recs1, recs2)):
                for c in cols:
                    v1 = r1.get(c)
                    v2 = r2.get(c)

                    # Null check
                    if (v1 is None or pd_isna(v1)) and (v2 is None or pd_isna(v2)):
                        continue
                    if (v1 is None or pd_isna(v1)) or (v2 is None or pd_isna(v2)):
                        raise AssertionError(
                            f"Values at row {i}, column '{c}' are different: {v1} != {v2}"
                        )

                    # Numeric comparison
                    try:
                        f1 = float(v1)
                        f2 = float(v2)
                        if not math.isclose(f1, f2, abs_tol=atol, rel_tol=rtol):
                            raise AssertionError(
                                f"Values at row {i}, column '{c}' are different: {v1} != {v2} (diff: {abs(f1-f2)})"
                            )
                    except (ValueError, TypeError):
                        if str(v1).strip().lower() != str(v2).strip().lower():
                            raise AssertionError(
                                f"Values at row {i}, column '{c}' are different: {v1} != {v2}"
                            )

    def pd_isna(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        if str(val).strip().lower() in ("nan", "none", "null", ""):
            return True
        return False

    def pd_to_numeric(series: Any, errors: str = "coerce") -> Series:
        if isinstance(series, Series):
            data = series._data
        elif isinstance(series, (list, tuple)):
            data = series
        else:
            data = [series]

        res = []
        for x in data:
            if x is None or pd_isna(x):
                res.append(float("nan"))
                continue
            try:
                res.append(float(x))
            except (ValueError, TypeError):
                if errors == "raise":
                    raise ValueError(f"Unable to parse string '{x}' as numeric")
                res.append(float("nan"))
        return Series(res, name=getattr(series, "name", None))

    def pd_to_datetime(series: Any, errors: str = "coerce") -> Series:
        return Series(series if hasattr(series, "__iter__") else [series])

    class _FakePandas:
        DataFrame = DataFrame
        Series = Series
        api = _PandasApi()
        testing = _Testing()
        isna = staticmethod(pd_isna)
        to_numeric = staticmethod(pd_to_numeric)
        to_datetime = staticmethod(pd_to_datetime)

    pd = _FakePandas()


# Unified exports
DataFrame = pd.DataFrame
Series = pd.Series
