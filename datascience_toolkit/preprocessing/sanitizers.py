"""
sanitizers.py
=============
Data type normalizers and missing value sanitizers.
Handles conversions between Athena/Pandas experimental types and Scikit-Learn standards.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def sanitize_athena_na(df: pd.DataFrame) -> pd.DataFrame:
  """Converts experimental pandas NA values (pd.NA) to IEEE NaN (np.nan)

  across object and string columns to maintain Scikit-Learn compatibility.
  """
  clean_df = df.copy()
  for col in clean_df.columns:
    if clean_df[col].dtype == "object" or isinstance(
        clean_df[col].dtype, pd.StringDtype
    ):
      clean_df[col] = clean_df[col].astype(object)
      clean_df.loc[clean_df[col].isna(), col] = np.nan
  return clean_df


def standardize_types(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    fill_numeric_zero: Optional[List[str]] = None,
) -> pd.DataFrame:
  """Normalizes column data types into standard float64 and object/category.

  Safely coerces errors into np.nan.
  """
  clean_df = sanitize_athena_na(df)

  if fill_numeric_zero:
    for col in fill_numeric_zero:
      if col in clean_df.columns:
        clean_df[col] = (
            pd.to_numeric(clean_df[col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

  if numeric_columns:
    for col in numeric_columns:
      if col in clean_df.columns:
        clean_df[col] = pd.to_numeric(
            clean_df[col], errors="coerce"
        ).astype(float)

  if categorical_columns:
    for col in categorical_columns:
      if col in clean_df.columns:
        clean_df[col] = clean_df[col].astype(object)
        clean_df.loc[clean_df[col].isna(), col] = np.nan

  return clean_df