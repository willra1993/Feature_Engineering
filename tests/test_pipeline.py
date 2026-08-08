import pandas as pd

from feature_engineering.pipeline import build_pipeline, prepare_features


def make_dataset(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_CLIENTE": range(rows),
            "TIPO_ENVIO_APLICACAO": ["Web" if index % 2 else "Carga" for index in range(rows)],
            "SEXO": ["F" if index % 3 else "M" for index in range(rows)],
            "FLAG_EMAIL": ["Y" if index % 2 else "N" for index in range(rows)],
            "RENDA_PESSOAL_MENSAL": [1_000 + 25 * index for index in range(rows)],
            "IDADE": [20 + index % 40 for index in range(rows)],
            "MAU_PAGADOR": [index % 2 for index in range(rows)],
        }
    )


def test_prepare_features_separates_target_and_drops_identifier() -> None:
    features, target = prepare_features(make_dataset())

    assert "MAU_PAGADOR" not in features
    assert "ID_CLIENTE" not in features
    assert set(target.unique()) == {0, 1}
    assert set(features["FLAG_EMAIL"].unique()) == {0.0, 1.0}


def test_pipeline_handles_missing_and_unknown_categories() -> None:
    features, target = prepare_features(make_dataset())
    train = features.iloc[:30].copy()
    test = features.iloc[30:].copy()
    test.loc[test.index[0], "TIPO_ENVIO_APLICACAO"] = "Telefone"
    test.loc[test.index[1], "RENDA_PESSOAL_MENSAL"] = pd.NA

    pipeline = build_pipeline(train)
    pipeline.fit(train, target.iloc[:30])
    predictions = pipeline.predict(test)

    assert len(predictions) == len(test)
    assert set(predictions).issubset({0, 1})
