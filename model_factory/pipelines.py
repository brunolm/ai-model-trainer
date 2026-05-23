from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


def build_pipeline(config, training_frame, feature_columns):
    task = config["task"]
    estimator = build_estimator(task, config.get("model") or {})

    if task == "text_classification":
        return Pipeline([
            ("features", build_text_vectorizer(config.get("features") or {})),
            ("model", estimator),
        ])

    return Pipeline([
        ("features", build_tabular_preprocessor(training_frame, feature_columns, config.get("features") or {})),
        ("model", estimator),
    ])


def build_text_vectorizer(features_config):
    if features_config.get("type", "tfidf") != "tfidf":
        raise ValueError("Only tfidf features are currently supported for text tasks")

    params = {
        "lowercase": features_config.get("lowercase", True),
        "analyzer": features_config.get("analyzer", "word"),
    }

    if features_config.get("ngram_range"):
        params["ngram_range"] = tuple(features_config["ngram_range"])

    for key in ["min_df", "max_df", "max_features", "stop_words"]:
        if key in features_config:
            params[key] = features_config[key]

    return TfidfVectorizer(**params)


def build_tabular_preprocessor(training_frame, feature_columns, features_config):
    numeric_columns, categorical_columns = split_tabular_columns(training_frame, feature_columns, features_config)
    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy=features_config.get("numeric_impute_strategy", "median"))),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy=features_config.get("categorical_impute_strategy", "most_frequent"))),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found")

    return ColumnTransformer(transformers=transformers)


def split_tabular_columns(training_frame, feature_columns, features_config):
    explicit_numeric = features_config.get("numeric_columns")
    explicit_categorical = features_config.get("categorical_columns")

    if explicit_numeric or explicit_categorical:
        numeric_columns = explicit_numeric or []
        categorical_columns = explicit_categorical or []
        validate_tabular_columns(feature_columns, numeric_columns, categorical_columns)
        return numeric_columns, categorical_columns

    numeric_columns = [
        column for column in feature_columns
        if training_frame[column].dtype.kind in {"i", "u", "f", "b"}
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def validate_tabular_columns(feature_columns, numeric_columns, categorical_columns):
    configured_columns = numeric_columns + categorical_columns
    unknown_columns = [column for column in configured_columns if column not in feature_columns]

    if unknown_columns:
        raise ValueError(f"Configured tabular columns are not feature columns: {unknown_columns}")

    duplicated_columns = set(numeric_columns).intersection(categorical_columns)

    if duplicated_columns:
        raise ValueError(f"Columns cannot be both numeric and categorical: {sorted(duplicated_columns)}")


def build_estimator(task, model_config):
    model_type = model_config.get("type") or default_model_type(task)
    params = model_config.get("params") or {}
    estimator_class, default_params = estimator_definition(task, model_type)
    return estimator_class(**{**default_params, **params})


def default_model_type(task):
    defaults = {
        "text_classification": "logistic_regression",
        "tabular_classification": "random_forest_classifier",
        "tabular_regression": "random_forest_regressor",
    }
    return defaults[task]


def estimator_definition(task, model_type):
    classifiers = {
        "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
        "linear_svc": (LinearSVC, {}),
        "multinomial_nb": (MultinomialNB, {}),
        "random_forest_classifier": (RandomForestClassifier, {"n_estimators": 200, "random_state": 42}),
        "gradient_boosting_classifier": (GradientBoostingClassifier, {"random_state": 42}),
    }
    regressors = {
        "linear_regression": (LinearRegression, {}),
        "random_forest_regressor": (RandomForestRegressor, {"n_estimators": 200, "random_state": 42}),
        "gradient_boosting_regressor": (GradientBoostingRegressor, {"random_state": 42}),
    }

    if task in {"text_classification", "tabular_classification"} and model_type in classifiers:
        return classifiers[model_type]

    if task == "tabular_regression" and model_type in regressors:
        return regressors[model_type]

    raise ValueError(f"Model type {model_type} is not supported for task {task}")
