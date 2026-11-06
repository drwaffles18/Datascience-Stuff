"""
pipeline.py
===========
Central preprocessing engine (DFPreprocess) for automated feature scaling,
imputation, and encoding.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .sanitizers import sanitize_athena_na


class DFPreprocess:
  """Automated tabular data preprocessor supporting imputation, scaling, and categorical encoding."""

  def __init__(
      self,
      scale_num: str = "standard",
      num_impute: str = "median",
      cat_impute: str = "most_frequent",
      encoding: str = "ordinal",
  ):
    self.scale_num = scale_num
    self.num_impute = num_impute
    self.cat_impute = cat_impute
    self.encoding = encoding
    self.pipeline: Optional[ColumnTransformer] = None
    self.feature_names_out_: List[str] = []
    self.numeric_cols_: List[str] = []
    self.categorical_cols_: List[str] = []

  def _get_scaler(self):
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    return scalers.get(self.scale_num, StandardScaler())

  def _get_encoder(self):
    if self.encoding == "onehot":
      return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

  def fit(
      self,
      df: pd.DataFrame,
      numeric_cols: Optional[List[str]] = None,
      categorical_cols: Optional[List[str]] = None,
  ):
    """Fits the imputers, scalers, and encoders on the provided DataFrame."""
    clean_df = sanitize_athena_na(df)

    if numeric_cols is None:
      self.numeric_cols_ = clean_df.select_dtypes(
          include=[np.number]
      ).columns.tolist()
    else:
      self.numeric_cols_ = [c for c in numeric_cols if c in clean_df.columns]

    if categorical_cols is None:
      self.categorical_cols_ = clean_df.select_dtypes(
          exclude=[np.number]
      ).columns.tolist()
    else:
      self.categorical_cols_ = [
          c for c in categorical_cols if c in clean_df.columns
      ]

    num_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=self.num_impute, keep_empty_features=True),
            ),
            ("scaler", self._get_scaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=self.cat_impute, fill_value="missing"
                ),
            ),
            ("encoder", self._get_encoder()),
        ]
    )

    transformers = []
    if self.numeric_cols_:
      transformers.append(("num", num_transformer, self.numeric_cols_))
    if self.categorical_cols_:
      transformers.append(("cat", cat_transformer, self.categorical_cols_))

    self.pipeline = ColumnTransformer(
        transformers=transformers, remainder="drop"
    )
    self.pipeline.fit(clean_df)

    self.feature_names_out_ = self.numeric_cols_ + self.categorical_cols_
    return self

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the DataFrame using the fitted preprocessors and returns a clean DataFrame."""
    if self.pipeline is None:
      raise ValueError("DFPreprocess instance is not fitted yet. Call fit().")

    clean_df = sanitize_athena_na(df)
    transformed_matrix = self.pipeline.transform(clean_df)

    if self.encoding == "onehot" and self.categorical_cols_:
      cat_encoder = self.pipeline.named_transformers_["cat"].named_steps[
          "encoder"
      ]
      cat_names = cat_encoder.get_feature_names_out(
          self.categorical_cols_
      ).tolist()
      cols_out = self.numeric_cols_ + cat_names
    else:
      cols_out = self.feature_names_out_

    return pd.DataFrame(
        transformed_matrix, columns=cols_out, index=clean_df.index
    )

  def fit_transform(
      self,
      df: pd.DataFrame,
      numeric_cols: Optional[List[str]] = None,
      categorical_cols: Optional[List[str]] = None,
  ) -> pd.DataFrame:
    return self.fit(
        df, numeric_cols=numeric_cols, categorical_cols=categorical_cols
    ).transform(df)