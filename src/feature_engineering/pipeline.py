"""Data loading and leakage-safe preprocessing for the credit-risk baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TARGET_COLUMN = "MAU_PAGADOR"

COLUMNS = [
    "ID_CLIENTE",
    "TIPO_FUNCIONARIO",
    "DIA_PAGAMENTO",
    "TIPO_ENVIO_APLICACAO",
    "QUANT_CARTOES_ADICIONAIS",
    "TIPO_ENDERECO_POSTAL",
    "SEXO",
    "ESTADO_CIVIL",
    "QUANT_DEPENDENTES",
    "NIVEL_EDUCACIONAL",
    "ESTADO_NASCIMENTO",
    "CIDADE_NASCIMENTO",
    "NACIONALIDADE",
    "ESTADO_RESIDENCIAL",
    "CIDADE_RESIDENCIAL",
    "BAIRRO_RESIDENCIAL",
    "FLAG_TELEFONE_RESIDENCIAL",
    "CODIGO_AREA_TELEFONE_RESIDENCIAL",
    "TIPO_RESIDENCIA",
    "MESES_RESIDENCIA",
    "FLAG_TELEFONE_MOVEL",
    "FLAG_EMAIL",
    "RENDA_PESSOAL_MENSAL",
    "OUTRAS_RENDAS",
    "FLAG_VISA",
    "FLAG_MASTERCARD",
    "FLAG_DINERS",
    "FLAG_AMERICAN_EXPRESS",
    "FLAG_OUTROS_CARTOES",
    "QUANT_CONTAS_BANCARIAS",
    "QUANT_CONTAS_BANCARIAS_ESPECIAIS",
    "VALOR_PATRIMONIO_PESSOAL",
    "QUANT_CARROS",
    "EMPRESA",
    "ESTADO_PROFISSIONAL",
    "CIDADE_PROFISSIONAL",
    "BAIRRO_PROFISSIONAL",
    "FLAG_TELEFONE_PROFISSIONAL",
    "CODIGO_AREA_TELEFONE_PROFISSIONAL",
    "MESES_NO_TRABALHO",
    "CODIGO_PROFISSAO",
    "TIPO_OCUPACAO",
    "CODIGO_PROFISSAO_CONJUGE",
    "NIVEL_EDUCACIONAL_CONJUGE",
    "FLAG_DOCUMENTO_RESIDENCIAL",
    "FLAG_RG",
    "FLAG_CPF",
    "FLAG_COMPROVANTE_RENDA",
    "PRODUTO",
    "FLAG_REGISTRO_ACSP",
    "IDADE",
    "CEP_RESIDENCIAL_3",
    "CEP_PROFISSIONAL_3",
    TARGET_COLUMN,
]

DROP_COLUMNS = [
    "ID_CLIENTE",
    "TIPO_FUNCIONARIO",
    "QUANT_CARTOES_ADICIONAIS",
    "NIVEL_EDUCACIONAL",
    "FLAG_TELEFONE_MOVEL",
    "FLAG_DOCUMENTO_RESIDENCIAL",
    "FLAG_RG",
    "FLAG_CPF",
    "FLAG_COMPROVANTE_RENDA",
    "FLAG_REGISTRO_ACSP",
    "CIDADE_NASCIMENTO",
    "CIDADE_RESIDENCIAL",
    "BAIRRO_RESIDENCIAL",
    "CIDADE_PROFISSIONAL",
    "BAIRRO_PROFISSIONAL",
    "CODIGO_AREA_TELEFONE_PROFISSIONAL",
    "CODIGO_AREA_TELEFONE_RESIDENCIAL",
    "CEP_RESIDENCIAL_3",
    "CEP_PROFISSIONAL_3",
]

BOOLEAN_COLUMNS = [
    "FLAG_TELEFONE_RESIDENCIAL",
    "FLAG_EMAIL",
    "FLAG_VISA",
    "FLAG_MASTERCARD",
    "FLAG_DINERS",
    "FLAG_AMERICAN_EXPRESS",
    "FLAG_TELEFONE_PROFISSIONAL",
    "EMPRESA",
]

CATEGORICAL_COLUMNS = [
    "DIA_PAGAMENTO",
    "TIPO_ENVIO_APLICACAO",
    "TIPO_ENDERECO_POSTAL",
    "SEXO",
    "ESTADO_CIVIL",
    "ESTADO_NASCIMENTO",
    "NACIONALIDADE",
    "ESTADO_RESIDENCIAL",
    "TIPO_RESIDENCIA",
    "FLAG_OUTROS_CARTOES",
    "ESTADO_PROFISSIONAL",
    "CODIGO_PROFISSAO",
    "TIPO_OCUPACAO",
    "CODIGO_PROFISSAO_CONJUGE",
    "NIVEL_EDUCACIONAL_CONJUGE",
    "PRODUTO",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the headerless, tab-separated source dataset with a stable schema."""
    data = pd.read_csv(
        path,
        sep="\t",
        names=COLUMNS,
        encoding="latin-1",
        low_memory=False,
        na_values=["NULL", " ", "#DIV/0!"],
    )
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, received {data.shape[1]}.")
    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize known values and separate predictors from the binary target."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}.")

    prepared = data.copy()
    prepared = prepared.replace({"NULL": pd.NA, "#DIV/0!": pd.NA, " ": pd.NA})

    boolean_map = {"Y": 1.0, "N": 0.0, "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0}
    for column in BOOLEAN_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(boolean_map)

    target = pd.to_numeric(prepared.pop(TARGET_COLUMN), errors="raise").astype("int8")
    invalid_targets = sorted(set(target.unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target must be binary; received: {invalid_targets}.")

    prepared = prepared.drop(columns=DROP_COLUMNS, errors="ignore")
    for column in CATEGORICAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("object")

    return prepared, target.rename(TARGET_COLUMN)


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and classification pipeline from the feature schema."""
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features.columns]
    numeric = [column for column in features.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
