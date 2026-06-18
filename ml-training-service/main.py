import base64
import io
import os
import random
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def supabase_headers(extra=None):
    headers = {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def rest_get(path, params=None):
    res = requests.get(f"{SUPABASE_URL}/rest/v1/{path}", headers=supabase_headers(), params=params, timeout=30)
    res.raise_for_status()
    return res.json()


def rest_post(path, payload, params=None, prefer="return=representation"):
    headers = supabase_headers({"Prefer": prefer})
    res = requests.post(f"{SUPABASE_URL}/rest/v1/{path}", headers=headers, params=params, json=payload, timeout=30)
    res.raise_for_status()
    return res.json()


def verify_user(access_token):
    res = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SERVICE_ROLE_KEY, "Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    if res.status_code != 200:
        return None
    return res.json()


def get_single(path, params):
    rows = rest_get(path, params=params)
    return rows[0] if rows else None


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if pd.isna(value):
        return None
    return value


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def is_missing_target(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "?", "na", "n/a", "null", "none", "unknown"}:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def rows_to_frame(rows, target_column):
    frame = pd.DataFrame(rows)
    if target_column not in frame.columns:
        raise ValueError("Target column is missing from split rows.")
    keep_mask = ~frame[target_column].map(is_missing_target)
    dropped = int((~keep_mask).sum())
    return frame.loc[keep_mask].copy(), dropped


def classify_columns(frame, feature_columns, column_summaries, excluded):
    summary_by_name = {item.get("name"): item for item in column_summaries or []}
    numeric, categorical, boolean, dropped = [], [], [], []
    for column in feature_columns:
        if column in excluded or column not in frame.columns:
            dropped.append(column)
            continue
        summary = summary_by_name.get(column, {})
        detected = summary.get("detected_type")
        if detected in {"datetime", "text", "unknown"} or summary.get("is_id_like") or summary.get("is_long_text"):
            dropped.append(column)
        elif detected == "numeric" or pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        elif detected == "boolean" or pd.api.types.is_bool_dtype(frame[column]):
            boolean.append(column)
        else:
            categorical.append(column)
    return numeric, categorical, boolean, dropped


def build_preprocessor(numeric_columns, categorical_columns, boolean_columns):
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns))
    cat_bool = categorical_columns + boolean_columns
    if cat_bool:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", one_hot_encoder())]), cat_bool))
    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models(task_type):
    if task_type == "tabular_classification":
        return [
            ("dummy_most_frequent", "Most frequent baseline", "DummyClassifier", DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}),
            ("logistic_regression", "Logistic regression", "LogisticRegression", LogisticRegression(max_iter=1000), {"max_iter": 1000}),
            ("random_forest", "Random forest", "RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42), {"n_estimators": 50, "random_state": 42}),
        ]
    return [
        ("dummy_mean", "Mean baseline", "DummyRegressor", DummyRegressor(strategy="mean"), {"strategy": "mean"}),
        ("ridge", "Ridge regression", "Ridge", Ridge(), {}),
        ("random_forest", "Random forest", "RandomForestRegressor", RandomForestRegressor(n_estimators=50, random_state=42), {"n_estimators": 50, "random_state": 42}),
    ]


def classification_primary(y_train):
    counts = pd.Series(y_train).value_counts().to_dict()
    if not counts:
        return "accuracy", counts
    values = list(counts.values())
    smallest, largest = min(values), max(values)
    return ("f1_macro" if largest and smallest / largest < 0.5 else "accuracy"), counts


def model_explanation(model_type, task_type):
    if "Dummy" in model_type:
        return "Simple baseline used as a sanity check.", ["Fast", "Easy to compare against"], ["Usually not accurate enough"]
    if model_type in {"LogisticRegression", "Ridge"}:
        return "Linear model that works well for many tabular datasets.", ["Fast", "Usually stable"], ["May miss complex patterns"]
    return "Tree-based model that can learn non-linear patterns.", ["Can capture complex patterns", "Strong baseline"], ["Less transparent", "Can be larger"]


def train_and_evaluate(task_type, preprocessor, x_train, y_train, x_val, y_val):
    results = []
    fitted = []
    primary_metric, class_distribution = (classification_primary(y_train) if task_type == "tabular_classification" else ("mae", {}))

    for model_id, name, model_type, estimator, params in make_models(task_type):
        started = time.perf_counter()
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_val)
            elapsed = int((time.perf_counter() - started) * 1000)
            if task_type == "tabular_classification":
                metrics = {
                    "accuracy": accuracy_score(y_val, predictions),
                    "f1_macro": f1_score(y_val, predictions, average="macro", zero_division=0),
                    "f1_weighted": f1_score(y_val, predictions, average="weighted", zero_division=0),
                    "class_distribution": class_distribution,
                }
            else:
                mae = mean_absolute_error(y_val, predictions)
                rmse = float(np.sqrt(np.mean((np.asarray(y_val, dtype=float) - np.asarray(predictions, dtype=float)) ** 2)))
                metrics = {"mae": mae, "rmse": rmse, "r2": r2_score(y_val, predictions)}
            explanation, pros, cons = model_explanation(model_type, task_type)
            sample = [{"actual": clean_json(a), "predicted": clean_json(p)} for a, p in list(zip(y_val, predictions))[:10]]
            result = {
                "model_id": model_id,
                "model_name": name,
                "model_type": model_type,
                "task_type": task_type,
                "status": "completed",
                "metrics": clean_json(metrics),
                "primary_metric_name": primary_metric,
                "primary_metric_value": clean_json(metrics[primary_metric]),
                "training_time_ms": elapsed,
                "warnings": [],
                "explanation": explanation,
                "pros": pros,
                "cons": cons,
                "model_parameters": params,
                "validation_predictions_sample": sample,
            }
            results.append(result)
            fitted.append((result, pipeline))
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            explanation, pros, cons = model_explanation(model_type, task_type)
            results.append({
                "model_id": model_id,
                "model_name": name,
                "model_type": model_type,
                "task_type": task_type,
                "status": "failed",
                "metrics": {},
                "primary_metric_name": primary_metric,
                "primary_metric_value": None,
                "training_time_ms": elapsed,
                "warnings": [str(exc)],
                "explanation": explanation,
                "pros": pros,
                "cons": cons,
                "model_parameters": params,
                "validation_predictions_sample": [],
            })
    if not fitted:
        raise ValueError("No models could be trained.")
    if task_type == "tabular_classification":
        fitted.sort(key=lambda item: (item[0]["primary_metric_value"], -item[0]["training_time_ms"]), reverse=True)
        reason = f"Recommended because it had the highest {primary_metric} on the validation split."
    else:
        fitted.sort(key=lambda item: (item[0]["metrics"].get("mae", float("inf")), item[0]["metrics"].get("rmse", float("inf")), item[0]["training_time_ms"]))
        reason = "Recommended because it had the lowest average error on the validation split."
    return results, fitted[0][0], fitted[0][1], reason


app = FastAPI(title="MLP Training Service")

allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
use_permissive_cors = not allowed_origins or "*" in allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if use_permissive_cors else allowed_origins,
    allow_origin_regex=None if use_permissive_cors else r"https://.*\.vercel\.app|http://localhost(:\d+)?",
    allow_credentials=not use_permissive_cors,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class TrainModelsRequest(BaseModel):
    pipe_id: str
    target_config_artifact_id: str


class ReviewResultsRequest(BaseModel):
    pipe_id: str
    trained_models_artifact_id: str


class PredictionSchemaRequest(BaseModel):
    pipe_id: str
    review_results_artifact_id: str


class TestPredictionSampleRequest(BaseModel):
    pipe_id: str
    review_results_artifact_id: str
    exclude_validation_row_indices: list[int] | None = None


class TestPredictionRequest(BaseModel):
    pipe_id: str
    review_results_artifact_id: str
    input: dict[str, Any]
    sample_context: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


def require_config():
    if not SUPABASE_URL or not SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Training service is missing server configuration.")


def require_user(authorization: str | None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token.")
    user = verify_user(authorization.replace("Bearer ", "", 1).strip())
    if not user or not user.get("id"):
        raise HTTPException(status_code=401, detail="Invalid Supabase session.")
    return user


@app.post("/train-models")
def train_models(request: TrainModelsRequest, authorization: str | None = Header(default=None)):
    require_config()
    user = require_user(authorization)

    try:
        pipe = get_single("pipes", {"id": f"eq.{request.pipe_id}", "select": "id,owner_id"})
        if not pipe or pipe.get("owner_id") != user["id"]:
            raise HTTPException(status_code=403, detail="You do not have access to train this pipe.")

        target_artifact = get_single(
            "artifacts",
            {"id": f"eq.{request.target_config_artifact_id}", "pipe_id": f"eq.{request.pipe_id}", "select": "id,content"},
        )
        if not target_artifact:
            raise HTTPException(status_code=404, detail="Target configuration artifact was not found.")
        target_config = target_artifact.get("content") or {}
        split_artifact_id = target_config.get("previous_split_dataset_artifact_id")
        split_artifact = get_single("artifacts", {"id": f"eq.{split_artifact_id}", "pipe_id": f"eq.{request.pipe_id}", "select": "id,content"})
        if not split_artifact:
            raise HTTPException(status_code=404, detail="Split dataset artifact was not found.")

        split_content = split_artifact.get("content") or {}
        train_rows = (split_content.get("splits") or {}).get("train") or []
        validation_rows = (split_content.get("splits") or {}).get("validation") or []
        target_column = target_config.get("target_column")
        task_type = target_config.get("detected_task_type")
        feature_columns = target_config.get("feature_columns") or []
        excluded_columns = set(target_config.get("excluded_feature_columns") or [])
        if task_type not in {"tabular_classification", "tabular_regression"}:
            raise HTTPException(status_code=400, detail="Unsupported task type.")
        if not train_rows or not validation_rows:
            raise HTTPException(status_code=400, detail="Train and validation splits are required.")

        train_df, dropped_train = rows_to_frame(train_rows, target_column)
        val_df, dropped_val = rows_to_frame(validation_rows, target_column)
        if train_df.empty or val_df.empty:
            raise HTTPException(status_code=400, detail="Not enough rows remain after dropping missing targets.")

        numeric_cols, categorical_cols, boolean_cols, dropped_cols = classify_columns(
            train_df,
            feature_columns,
            target_config.get("column_summaries") or [],
            excluded_columns,
        )
        usable_features = numeric_cols + categorical_cols + boolean_cols
        preprocessor = build_preprocessor(numeric_cols, categorical_cols, boolean_cols)
        x_train = train_df[usable_features]
        x_val = val_df[usable_features]
        y_train = train_df[target_column]
        y_val = val_df[target_column]
        if task_type == "tabular_regression":
            y_train = pd.to_numeric(y_train, errors="coerce")
            y_val = pd.to_numeric(y_val, errors="coerce")
            keep_train = ~y_train.isna()
            keep_val = ~y_val.isna()
            dropped_train += int((~keep_train).sum())
            dropped_val += int((~keep_val).sum())
            x_train, y_train = x_train.loc[keep_train], y_train.loc[keep_train]
            x_val, y_val = x_val.loc[keep_val], y_val.loc[keep_val]

        models, recommended, fitted_pipeline, recommendation_reason = train_and_evaluate(task_type, preprocessor, x_train, y_train, x_val, y_val)
        bundle_io = io.BytesIO()
        # TODO: Move serialized model bundles to Supabase Storage before alpha.
        joblib.dump(fitted_pipeline, bundle_io)
        bundle = base64.b64encode(bundle_io.getvalue()).decode("ascii")
        training_summary = {
            "train_rows_total": len(train_rows),
            "train_rows_used": int(len(x_train)),
            "validation_rows_total": len(validation_rows),
            "validation_rows_used": int(len(x_val)),
            "dropped_train_rows_missing_target": dropped_train,
            "dropped_validation_rows_missing_target": dropped_val,
            "feature_count_after_preprocessing": len(usable_features),
        }
        content = clean_json({
            "previous_target_config_artifact_id": request.target_config_artifact_id,
            "previous_split_dataset_artifact_id": split_artifact_id,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "excluded_feature_columns": list(excluded_columns),
            "preprocessing": {
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "boolean_columns": boolean_cols,
                "dropped_columns": dropped_cols,
            },
            "models": models,
            "recommended_model_id": recommended["model_id"],
            "recommended_model_name": recommended["model_name"],
            "recommended_model_bundle": {"format": "joblib_base64", "value": bundle},
            "recommendation_reason": recommendation_reason,
            "training_summary": training_summary,
        })
        artifact_payload = {
            "pipe_id": request.pipe_id,
            "artifact_type": "trained_models",
            "kind": "trained_models",
            "name": "Trained models",
            "content": content,
            "metadata": {
                "previous_target_config_artifact_id": request.target_config_artifact_id,
                "task_type": task_type,
                "target_column": target_column,
                "recommended_model_id": recommended["model_id"],
                "recommended_model_name": recommended["model_name"],
                "primary_metric_name": recommended["primary_metric_name"],
                "primary_metric_value": recommended["primary_metric_value"],
                "trained_model_count": len([m for m in models if m.get("status") == "completed"]),
            },
        }
        artifact = rest_post("artifacts", artifact_payload)[0]
        output = {
            "step_key": "train_models",
            "status": "completed",
            "trained_models_artifact_id": artifact["id"],
            "previous_target_config_artifact_id": request.target_config_artifact_id,
            "task_type": task_type,
            "target_column": target_column,
            "recommended_model_id": recommended["model_id"],
            "recommended_model_name": recommended["model_name"],
            "primary_metric_name": recommended["primary_metric_name"],
            "primary_metric_value": recommended["primary_metric_value"],
            "model_count": len([m for m in models if m.get("status") == "completed"]),
            "storage": {"format": "json", "uri": f"artifact:{artifact['id']}"},
        }
        rest_post(
            "pipe_step_outputs",
            {"pipe_id": request.pipe_id, "step_key": "train_models", "artifact_id": artifact["id"], "status": "completed", "output": output},
            params={"on_conflict": "pipe_id,step_key"},
            prefer="resolution=merge-duplicates,return=representation",
        )
        return {
            "trained_models_artifact_id": artifact["id"],
            "recommended_model_id": recommended["model_id"],
            "recommended_model_name": recommended["model_name"],
            "primary_metric_name": recommended["primary_metric_name"],
            "primary_metric_value": recommended["primary_metric_value"],
            "model_count": output["model_count"],
            "task_type": task_type,
            "target_column": target_column,
        }
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def load_owned_pipe(pipe_id: str, user_id: str):
    pipe = get_single("pipes", {"id": f"eq.{pipe_id}", "select": "id,owner_id"})
    if not pipe or pipe.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this pipe.")
    return pipe


def load_artifact(pipe_id: str, artifact_id: str, select: str = "id,content,metadata"):
    artifact = get_single("artifacts", {"id": f"eq.{artifact_id}", "pipe_id": f"eq.{pipe_id}", "select": select})
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    return artifact


def decode_recommended_pipeline(bundle: dict[str, Any]):
    if not bundle or bundle.get("format") != "joblib_base64" or not bundle.get("value"):
        raise ValueError("Recommended model bundle is missing from the training artifact.")
    raw = base64.b64decode(bundle["value"])
    return joblib.load(io.BytesIO(raw))


def figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def chart_payload(
    chart_key: str,
    title: str,
    kind: str,
    fig,
    description: str,
    how_to_read: str = "",
    why_it_matters: str = "",
    caveats: list[str] | None = None,
    uses_all_validation_rows: bool = True,
    shows_actual_labels: bool = False,
    shows_model_predictions: bool = False,
    shows_prediction_errors: bool = False,
    shows_sample_only: bool = False,
):
    return {
        "chart_key": chart_key,
        "title": title,
        "kind": kind,
        "image_format": "png_base64",
        "image_base64": figure_to_base64(fig),
        "description": description,
        "how_to_read": how_to_read or description,
        "why_it_matters": why_it_matters,
        "caveats": caveats or [],
        "uses_all_validation_rows": uses_all_validation_rows,
        "shows_actual_labels": shows_actual_labels,
        "shows_model_predictions": shows_model_predictions,
        "shows_prediction_errors": shows_prediction_errors,
        "shows_sample_only": shows_sample_only,
    }


def model_comparison_chart(models: list[dict[str, Any]], task_type: str):
    completed = [m for m in models if m.get("status") == "completed" and m.get("primary_metric_value") is not None]
    if not completed:
        return None
    labels = [m.get("model_name", "Model") for m in completed]
    values = [float(m.get("primary_metric_value") or 0) for m in completed]
    metric = completed[0].get("primary_metric_name") or ("mae" if task_type == "tabular_regression" else "accuracy")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color="#111827", label=metric.replace("_", " "))
    ax.set_title("Model comparison")
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=18, labelsize=10)
    ax.legend(loc="best")
    return chart_payload(
        "model_comparison",
        "Model comparison",
        "bar",
        fig,
        "Secondary view. Compares the models that were actually trained using their validation metric.",
        how_to_read="Each bar is one trained model. Taller is better for classification metrics; lower-error regression models are summarized in the model table above.",
        why_it_matters="This checks whether the recommended model was clearly better than the alternatives.",
        caveats=["This repeats the metric comparison and should not be read as a separate evaluation."],
        shows_model_predictions=True,
    )


def classification_charts(y_true, y_pred, models):
    charts = []
    true_labels = [str(v) for v in y_true]
    pred_labels = [str(v) for v in y_pred]
    labels = sorted({*true_labels, *pred_labels})
    matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    row_totals = matrix.sum(axis=1)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), max(6, len(labels) * 1.2)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix", fontsize=16, pad=16)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("Actual class", fontsize=12)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pct = (matrix[i, j] / row_totals[i] * 100) if row_totals[i] else 0
            if matrix.shape == (2, 2):
                meaning = f"correct {labels[i]}" if i == j else f"predicted {labels[j]}\nactually {labels[i]}"
                text = f"{int(matrix[i, j])} rows\n{pct:.0f}%\n{meaning}"
            else:
                meaning = "correct" if i == j else "mistake"
                text = f"{int(matrix[i, j])} rows\n{pct:.0f}%\n{meaning}"
            ax.text(j, i, text, ha="center", va="center", color="#111827", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Validation rows")
    charts.append(chart_payload(
        "confusion_matrix",
        "Confusion matrix",
        "confusion_matrix",
        fig,
        "Rows are the real labels. Columns are the model predictions. Diagonal cells are correct predictions; the other cells are mistakes.",
        how_to_read="Rows are the real labels. Columns are the model predictions. The diagonal cells are correct predictions. The other cells are mistakes.",
        why_it_matters="This shows exactly where the model is right and where it confuses one class for another.",
        caveats=["Use this chart to understand mistakes, not just the overall score."],
        shows_actual_labels=True,
        shows_model_predictions=True,
        shows_prediction_errors=True,
    ))

    correct_by_actual = []
    wrong_by_actual = []
    for label in labels:
        actual_mask = np.asarray(true_labels) == label
        predicted_for_actual = np.asarray(pred_labels)[actual_mask]
        correct_by_actual.append(int((predicted_for_actual == label).sum()))
        wrong_by_actual.append(int((predicted_for_actual != label).sum()))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    ax.bar(x, correct_by_actual, label="Correct predictions", color="#047857")
    ax.bar(x, wrong_by_actual, bottom=correct_by_actual, label="Wrong predictions", color="#dc2626")
    ax.set_title("Prediction outcomes by actual class", fontsize=15, pad=12)
    ax.set_ylabel("Validation rows")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.legend(loc="best")
    charts.append(chart_payload(
        "prediction_outcomes_by_actual_class",
        "Prediction outcomes by actual class",
        "stacked_bar",
        fig,
        "Each bar starts from the real class. Green shows correct predictions and red shows mistakes.",
        how_to_read="Each bar starts from the real class. The segments show how the model classified those rows.",
        why_it_matters="This reveals whether the model struggles more with one class than another.",
        caveats=["Unlike simple actual-vs-predicted totals, this chart does not hide mistakes that cancel each other out."],
        shows_actual_labels=True,
        shows_model_predictions=True,
        shows_prediction_errors=True,
    ))

    comparison = model_comparison_chart(models, "tabular_classification")
    if comparison:
        charts.append(comparison)
    return charts


def regression_charts(y_true, y_pred, models):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = predicted - actual
    absolute_errors = np.abs(residuals)
    charts = []

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.72, color="#111827", label="Validation row")
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#dc2626", label="Perfect prediction")
    ax.set_title("Predicted vs actual", fontsize=15, pad=12)
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    ax.legend(loc="best")
    charts.append(chart_payload(
        "predicted_vs_actual",
        "Predicted vs actual",
        "scatter",
        fig,
        "Points close to the diagonal are better predictions.",
        how_to_read="Each point is one validation row. The diagonal line means a perfect prediction.",
        why_it_matters="This shows whether predictions generally follow the real target values.",
        caveats=["Outliers are not hidden; a few large errors can dominate the visual pattern."],
        shows_actual_labels=True,
        shows_model_predictions=True,
    ))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(predicted, residuals, alpha=0.72, color="#7c3aed", label="Prediction error")
    ax.axhline(0, linestyle="--", color="#111827", label="Zero error")
    ax.set_title("Residuals", fontsize=15, pad=12)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Prediction error (predicted − real)")
    ax.legend(loc="best")
    charts.append(chart_payload(
        "residuals",
        "Residuals",
        "scatter",
        fig,
        "Prediction error means predicted value minus real value. Points close to zero are better.",
        how_to_read="Each point is one validation row. Points above zero are over-predictions; points below zero are under-predictions.",
        why_it_matters="This helps spot whether the model is consistently too high or too low for some predictions.",
        caveats=["A few large errors can make the rest of the errors look smaller."],
        shows_actual_labels=True,
        shows_model_predictions=True,
        shows_prediction_errors=True,
    ))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(absolute_errors, bins=min(20, max(5, int(np.sqrt(len(absolute_errors))))), color="#2563eb", alpha=0.82, label="Absolute error")
    ax.set_title("Error distribution", fontsize=15, pad=12)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Validation rows")
    ax.legend(loc="best")
    charts.append(chart_payload(
        "error_distribution",
        "Error distribution",
        "histogram",
        fig,
        "Most errors should be close to zero.",
        how_to_read="Bars near zero mean many predictions were close to the real value. Bars far from zero are larger mistakes.",
        why_it_matters="This shows whether typical errors are small or whether a few large errors are common.",
        caveats=["The units match the target column, so large or small depends on what you are predicting."],
        shows_actual_labels=True,
        shows_model_predictions=True,
        shows_prediction_errors=True,
    ))

    comparison = model_comparison_chart(models, "tabular_regression")
    if comparison:
        charts.append(comparison)
    return charts


def feature_importance_chart(pipeline, x_val: pd.DataFrame, y_val, task_type: str, primary_metric: str):
    scoring = "neg_mean_absolute_error" if task_type == "tabular_regression" else ("f1_macro" if primary_metric == "f1_macro" else "accuracy")
    result = permutation_importance(pipeline, x_val, y_val, scoring=scoring, n_repeats=5, random_state=42)
    importances = sorted(
        [{"feature": column, "importance": float(value)} for column, value in zip(x_val.columns, result.importances_mean)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    top = importances[:8]
    if not top:
        return None, []
    labels = [item["feature"] for item in reversed(top)]
    values = [item["importance"] for item in reversed(top)]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.55)))
    bars = ax.barh(labels, values, color="#0f766e", label="Validation importance")
    ax.set_title("What influenced the model most?", fontsize=15, pad=12)
    ax.set_xlabel("Change in validation score when shuffled")
    ax.legend(loc="best")
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center", fontsize=9)
    chart = chart_payload(
        "feature_importance",
        "What influenced the model most?",
        "bar",
        fig,
        "These are the columns that changed the model's validation score the most when their values were shuffled.",
        how_to_read="Longer bars had a larger effect on the validation score when that column was shuffled.",
        why_it_matters="This helps explain which input columns the recommended model relied on most.",
        caveats=[
            "Importance does not prove causation.",
            "Correlated features can share importance.",
            "This is measured on validation data, not on future production data.",
        ],
        shows_actual_labels=True,
        shows_model_predictions=True,
    )
    return chart, importances


def looks_id_like(feature: str, numeric: pd.Series, total_rows: int):
    name = feature.lower().replace("_", "")
    unique_ratio = numeric.nunique(dropna=True) / max(1, numeric.notna().sum())
    return name == "id" or name.endswith("id") or (unique_ratio > 0.95 and numeric.notna().sum() > min(50, total_rows * 0.8))


def choose_numeric_feature(importances: list[dict[str, Any]], x_val: pd.DataFrame):
    ordered = [item.get("feature") for item in importances] or list(x_val.columns)
    minimum_non_null = min(20, max(5, len(x_val) // 3))
    for feature in ordered:
        if feature not in x_val.columns:
            continue
        numeric = pd.to_numeric(x_val[feature], errors="coerce")
        non_null = numeric.dropna()
        if len(non_null) < minimum_non_null:
            continue
        if non_null.nunique() < min(8, max(3, len(non_null) // 3)):
            continue
        if looks_id_like(str(feature), non_null, len(x_val)):
            continue
        most_common_share = float(non_null.value_counts(normalize=True).iloc[0]) if not non_null.empty else 1.0
        if most_common_share > 0.70:
            continue
        zero_share = float((non_null == 0).mean()) if len(non_null) else 1.0
        if zero_share > 0.70:
            continue
        q10, q90 = non_null.quantile(0.10), non_null.quantile(0.90)
        if q10 == q90:
            continue
        return feature, numeric
    return None, None


def positive_class_for_probabilities(pipeline, y_true):
    model = getattr(pipeline, "named_steps", {}).get("model")
    classes = [str(item) for item in getattr(model, "classes_", [])]
    if len(classes) != 2:
        return None, None
    preferred = [item for item in classes if ">" in item or item.lower() in {"true", "yes", "1", "positive", "churn"}]
    label = preferred[0] if preferred else classes[-1]
    return label, classes.index(label)


def feature_relationship_charts(pipeline, x_val: pd.DataFrame, y_val, y_pred, task_type: str, target_column: str, importances: list[dict[str, Any]]):
    feature, numeric_values = choose_numeric_feature(importances, x_val)
    if not feature or numeric_values is None:
        return [], "No numeric feature was varied enough to create a meaningful relationship chart."
    valid_mask = numeric_values.notna()
    x_feature = numeric_values.loc[valid_mask]
    x_subset = x_val.loc[valid_mask]
    y_subset = pd.Series(list(y_val), index=x_val.index).loc[valid_mask]
    pred_subset = pd.Series(list(y_pred), index=x_val.index).loc[valid_mask]
    charts = []
    if task_type == "tabular_classification" and hasattr(pipeline, "predict_proba"):
        positive_label, positive_index = positive_class_for_probabilities(pipeline, y_val)
        if positive_label is not None and positive_index is not None:
            probabilities = pipeline.predict_proba(x_subset)[:, positive_index]
            correctness = [str(a) == str(p) for a, p in zip(y_subset, pred_subset)]
            correct_mask = np.asarray(correctness)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.scatter(x_feature[correct_mask], probabilities[correct_mask], label="Correct prediction", color="#047857", alpha=0.72)
            ax.scatter(x_feature[~correct_mask], probabilities[~correct_mask], label="Wrong prediction", color="#dc2626", alpha=0.82)
            ax.set_title(f"How {feature} relates to the prediction", fontsize=15, pad=12)
            ax.set_xlabel(feature)
            ax.set_ylabel(f"Predicted probability of {positive_label}")
            ax.legend(loc="best")
            charts.append(chart_payload(
                "top_feature_relationship",
                f"How {feature} relates to the prediction",
                "scatter",
                fig,
                f"Each point is one validation row. Higher points mean the model is more confident in predicting {positive_label}.",
                how_to_read=f"Each point is one validation row. The x-axis is {feature}. The y-axis is how confident the model was that the row belongs to {positive_label}.",
                why_it_matters="This helps you see whether the model's confidence changes as this feature changes.",
                caveats=["This chart shows one feature at a time.", "The model may use many features together, so this is not a complete explanation of the model."],
                shows_actual_labels=True,
                shows_model_predictions=True,
                shows_prediction_errors=True,
            ))

            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.scatter(x_feature[correct_mask], probabilities[correct_mask], label="Correct prediction", color="#047857", alpha=0.68)
            ax.scatter(x_feature[~correct_mask], probabilities[~correct_mask], label="Wrong prediction", color="#dc2626", alpha=0.85)
            ax.set_xlabel(feature)
            ax.set_ylabel(f"Predicted probability of {positive_label}")
            ax.set_title("Where did the model make mistakes?", fontsize=15, pad=12)
            ax.legend(loc="best")
            charts.append(chart_payload(
                "mistakes_by_top_feature",
                "Where did the model make mistakes?",
                "scatter",
                fig,
                "Red points are validation rows the model predicted incorrectly.",
                how_to_read="Red points are validation rows the model predicted incorrectly. Green points were predicted correctly.",
                why_it_matters="If red points cluster in one area, the model may struggle with that type of example.",
                caveats=["This chart uses one feature as the x-axis, but mistakes can depend on multiple features together."],
                shows_actual_labels=True,
                shows_model_predictions=True,
                shows_prediction_errors=True,
            ))
    elif task_type == "tabular_regression":
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x_feature, pd.to_numeric(y_subset, errors="coerce"), alpha=0.62, label="Actual target", color="#047857")
        ax.scatter(x_feature, pd.to_numeric(pred_subset, errors="coerce"), alpha=0.62, label="Predicted target", color="#2563eb")
        ax.set_title(f"How {feature} relates to predicted {target_column}", fontsize=15, pad=12)
        ax.set_xlabel(feature)
        ax.set_ylabel(target_column)
        ax.legend(loc="best")
        charts.append(chart_payload(
            "top_feature_relationship",
            f"How {feature} relates to predicted {target_column}",
            "scatter",
            fig,
            "Green points are actual values and blue points are model predictions for the same feature values.",
            how_to_read=f"The x-axis is {feature}. Green points show real {target_column}; blue points show predicted {target_column}.",
            why_it_matters="This helps you see whether predictions move with an important numeric feature.",
            caveats=["This chart shows one feature at a time and does not hide outliers."],
            shows_actual_labels=True,
            shows_model_predictions=True,
        ))
    return charts, None


def review_notes(task_type: str, y_true, models: list[dict[str, Any]]):
    notes = []
    if len(y_true) < 30:
        notes.append("The validation set is small, so results may change with more data.")
    failed = [m.get("model_name") for m in models if m.get("status") == "failed"]
    if failed:
        notes.append(f"Some models could not be trained: {', '.join(failed)}.")
    completed = [m for m in models if m.get("status") == "completed" and m.get("primary_metric_value") is not None]
    if len(completed) >= 2:
        reverse = task_type == "tabular_classification"
        ordered = sorted(completed, key=lambda m: float(m.get("primary_metric_value") or 0), reverse=reverse)
        if task_type == "tabular_regression":
            ordered = sorted(completed, key=lambda m: float(m.get("metrics", {}).get("mae", float("inf"))))
        first = float(ordered[0].get("primary_metric_value") or 0)
        second = float(ordered[1].get("primary_metric_value") or 0)
        if first and abs(first - second) / abs(first) < 0.05:
            notes.append("The top models are close, so the recommendation is not dramatically better than the runner-up.")
    if task_type == "tabular_classification":
        counts = pd.Series(y_true).value_counts()
        if len(counts) > 1 and counts.min() / counts.max() < 0.5:
            notes.append("The validation data is imbalanced, so F1 score may be more useful than accuracy.")
    return notes


def plain_english_summary(task_type: str, recommended: dict[str, Any], notes: list[str]):
    name = recommended.get("model_name", "The recommended model")
    metric = recommended.get("primary_metric_name", "metric").replace("_", " ")
    value = recommended.get("primary_metric_value")
    metric_text = f" ({float(value):.3f})" if isinstance(value, (int, float)) and np.isfinite(value) else ""
    if task_type == "tabular_classification":
        summary = f"{name} is recommended because it achieved the best validation {metric}{metric_text} among the models tested. The confusion matrix shows which classes it predicts correctly and where mistakes happen."
    else:
        mae = (recommended.get("metrics") or {}).get("mae")
        error_text = f" On average, predictions are off by about {float(mae):.2f} units." if isinstance(mae, (int, float)) and np.isfinite(mae) else ""
        summary = f"{name} is recommended because it achieved the lowest average error on the validation data.{error_text} The predicted vs actual chart shows how closely predictions follow the real values."
    if notes:
        summary += " " + notes[0]
    return summary


def prediction_examples(val_df: pd.DataFrame, feature_columns: list[str], target_column: str, predictions, task_type: str):
    visible_features = [col for col in feature_columns if col in val_df.columns and col != target_column][:5]
    columns = visible_features + ["actual", "predicted"] + (["correct"] if task_type == "tabular_classification" else ["absolute_error"])
    rows = []
    for idx, (_, row) in enumerate(val_df.head(10).iterrows()):
        actual = row[target_column]
        predicted = predictions[idx]
        item = {col: clean_json(row[col]) for col in visible_features}
        item["actual"] = clean_json(actual)
        item["predicted"] = clean_json(predicted)
        if task_type == "tabular_classification":
            item["correct"] = str(actual) == str(predicted)
        else:
            try:
                item["absolute_error"] = clean_json(abs(float(actual) - float(predicted)))
            except Exception:
                item["absolute_error"] = None
        rows.append(item)
    return {"columns": columns, "rows": rows}


@app.post("/review-results")
def review_results(request: ReviewResultsRequest, authorization: str | None = Header(default=None)):
    require_config()
    user = require_user(authorization)

    try:
        load_owned_pipe(request.pipe_id, user["id"])
        trained_artifact = load_artifact(request.pipe_id, request.trained_models_artifact_id)
        trained_content = trained_artifact.get("content") or {}
        target_artifact_id = trained_content.get("previous_target_config_artifact_id")
        split_artifact_id = trained_content.get("previous_split_dataset_artifact_id")
        if not target_artifact_id or not split_artifact_id:
            raise HTTPException(status_code=400, detail="Training artifact is missing lineage metadata.")

        target_artifact = load_artifact(request.pipe_id, target_artifact_id)
        split_artifact = load_artifact(request.pipe_id, split_artifact_id)
        target_config = target_artifact.get("content") or {}
        split_content = split_artifact.get("content") or {}
        validation_rows = (split_content.get("splits") or {}).get("validation") or []
        if not validation_rows:
            raise HTTPException(status_code=400, detail="Validation split is missing or empty.")

        target_column = trained_content.get("target_column") or target_config.get("target_column")
        task_type = trained_content.get("task_type") or target_config.get("detected_task_type")
        feature_columns = trained_content.get("feature_columns") or target_config.get("feature_columns") or []
        excluded_columns = trained_content.get("excluded_feature_columns") or target_config.get("excluded_feature_columns") or []
        preprocessing = trained_content.get("preprocessing") or {}
        usable_features = (preprocessing.get("numeric_columns") or []) + (preprocessing.get("categorical_columns") or []) + (preprocessing.get("boolean_columns") or [])
        if not usable_features:
            usable_features = [col for col in feature_columns if col not in set(excluded_columns)]
        if task_type not in {"tabular_classification", "tabular_regression"} or not target_column:
            raise HTTPException(status_code=400, detail="Training artifact is missing target or task metadata.")

        val_df, dropped_val = rows_to_frame(validation_rows, target_column)
        missing_features = [col for col in usable_features if col not in val_df.columns]
        usable_features = [col for col in usable_features if col in val_df.columns]
        if not usable_features or val_df.empty:
            raise HTTPException(status_code=400, detail="Validation data is not usable for review.")
        y_val = val_df[target_column]
        if task_type == "tabular_regression":
            y_numeric = pd.to_numeric(y_val, errors="coerce")
            keep_val = ~y_numeric.isna()
            dropped_val += int((~keep_val).sum())
            val_df = val_df.loc[keep_val].copy()
            y_val = y_numeric.loc[keep_val]
        if val_df.empty:
            raise HTTPException(status_code=400, detail="No validation rows remain after removing missing targets.")

        pipeline = decode_recommended_pipeline(trained_content.get("recommended_model_bundle") or {})
        x_val = val_df[usable_features]
        predictions = pipeline.predict(x_val)
        models = trained_content.get("models") or []
        recommended_id = trained_content.get("recommended_model_id")
        recommended = next((m for m in models if m.get("model_id") == recommended_id), None) or {}
        if not recommended:
            recommended = {
                "model_id": recommended_id,
                "model_name": trained_content.get("recommended_model_name", "Recommended model"),
                "primary_metric_name": "mae" if task_type == "tabular_regression" else "accuracy",
                "primary_metric_value": None,
                "metrics": {},
                "explanation": trained_content.get("recommendation_reason", "Recommended from the validation results."),
                "pros": [],
                "cons": [],
                "warnings": [],
            }
        model_comparison = [{
            "model_id": m.get("model_id"),
            "model_name": m.get("model_name"),
            "primary_metric_name": m.get("primary_metric_name"),
            "primary_metric_value": m.get("primary_metric_value"),
            "metrics": m.get("metrics") or {},
            "training_time_ms": m.get("training_time_ms"),
            "status": m.get("status"),
        } for m in models]

        notes = review_notes(task_type, y_val, models)
        if missing_features:
            notes.append(f"Some saved feature columns were not present in validation data: {', '.join(missing_features[:5])}.")
        if dropped_val:
            notes.append(f"{dropped_val} validation rows were skipped because the target was missing.")
        if task_type == "tabular_classification":
            charts = classification_charts(y_val, predictions, models)
        else:
            charts = regression_charts(y_val, predictions, models)
        primary_metric = (recommended or {}).get("primary_metric_name") or ("mae" if task_type == "tabular_regression" else "accuracy")
        try:
            importance_chart, feature_importances = feature_importance_chart(pipeline, x_val, y_val, task_type, primary_metric)
            if importance_chart:
                charts.insert(2, importance_chart)
            relationship_charts, relationship_note = feature_relationship_charts(pipeline, x_val, y_val, predictions, task_type, target_column, feature_importances)
            charts[3:3] = relationship_charts
            if relationship_note:
                notes.append(relationship_note)
        except Exception as exc:
            notes.append(f"Feature importance charts could not be generated: {exc}.")

        recommended_model = {
            "model_id": recommended.get("model_id"),
            "model_name": recommended.get("model_name") or trained_content.get("recommended_model_name"),
            "primary_metric_name": recommended.get("primary_metric_name"),
            "primary_metric_value": recommended.get("primary_metric_value"),
            "metrics": recommended.get("metrics") or {},
            "explanation": recommended.get("explanation") or trained_content.get("recommendation_reason") or "Recommended from the validation results.",
            "pros": recommended.get("pros") or [],
            "cons": recommended.get("cons") or [],
            "warnings": recommended.get("warnings") or [],
        }
        content = clean_json({
            "previous_trained_models_artifact_id": request.trained_models_artifact_id,
            "task_type": task_type,
            "target_column": target_column,
            "recommended_model": recommended_model,
            "model_comparison": model_comparison,
            "plain_english_summary": plain_english_summary(task_type, recommended_model, notes),
            "validation_summary": {"rows_evaluated": int(len(val_df)), "notes": notes},
            "charts": charts,
            "prediction_examples": prediction_examples(val_df, feature_columns, target_column, predictions, task_type),
        })
        artifact_payload = {
            "pipe_id": request.pipe_id,
            "artifact_type": "review_results",
            "kind": "review_results",
            "name": "Review results",
            "content": content,
            "metadata": {
                "previous_trained_models_artifact_id": request.trained_models_artifact_id,
                "task_type": task_type,
                "target_column": target_column,
                "recommended_model_name": recommended_model.get("model_name"),
                "primary_metric_name": recommended_model.get("primary_metric_name"),
                "primary_metric_value": recommended_model.get("primary_metric_value"),
            },
        }
        artifact = rest_post("artifacts", artifact_payload)[0]
        output = {
            "step_key": "review_results",
            "status": "completed",
            "review_results_artifact_id": artifact["id"],
            "previous_trained_models_artifact_id": request.trained_models_artifact_id,
            "recommended_model_name": recommended_model.get("model_name"),
            "primary_metric_name": recommended_model.get("primary_metric_name"),
            "primary_metric_value": recommended_model.get("primary_metric_value"),
            "storage": {"format": "json", "uri": f"artifact:{artifact['id']}"},
        }
        rest_post(
            "pipe_step_outputs",
            {"pipe_id": request.pipe_id, "step_key": "review_results", "artifact_id": artifact["id"], "status": "completed", "output": output},
            params={"on_conflict": "pipe_id,step_key"},
            prefer="resolution=merge-duplicates,return=representation",
        )
        return {"review_results_artifact_id": artifact["id"], **content}
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def load_prediction_lineage(pipe_id: str, review_results_artifact_id: str):
    review_artifact = load_artifact(pipe_id, review_results_artifact_id)
    review_content = review_artifact.get("content") or {}
    trained_artifact_id = review_content.get("previous_trained_models_artifact_id")
    if not trained_artifact_id:
        raise HTTPException(status_code=400, detail="Review results artifact is missing trained model lineage.")
    trained_artifact = load_artifact(pipe_id, trained_artifact_id)
    trained_content = trained_artifact.get("content") or {}
    split_artifact = None
    split_artifact_id = trained_content.get("previous_split_dataset_artifact_id")
    if split_artifact_id:
        split_artifact = load_artifact(pipe_id, split_artifact_id)
    return review_artifact, trained_artifact_id, trained_content, split_artifact


def usable_prediction_columns(trained_content: dict[str, Any]):
    preprocessing = trained_content.get("preprocessing") or {}
    numeric = [col for col in preprocessing.get("numeric_columns") or [] if isinstance(col, str)]
    categorical = [col for col in preprocessing.get("categorical_columns") or [] if isinstance(col, str)]
    boolean = [col for col in preprocessing.get("boolean_columns") or [] if isinstance(col, str)]
    target_column = trained_content.get("target_column")
    excluded = set(trained_content.get("excluded_feature_columns") or [])
    dropped = set(preprocessing.get("dropped_columns") or [])
    seen = set()
    fields = []
    for column_type, columns in [("number", numeric), ("text", categorical), ("boolean", boolean)]:
        for name in columns:
            if name == target_column or name in excluded or name in dropped or name in seen:
                continue
            seen.add(name)
            fields.append({"name": name, "type": column_type})
    return fields


def example_for_column(rows: list[dict[str, Any]], column: str, column_type: str):
    values = [row.get(column) for row in rows if isinstance(row, dict) and not is_missing_target(row.get(column))]
    if not values:
        return None
    if column_type == "number":
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if numeric.empty:
            return None
        return clean_json(float(numeric.median()))
    if column_type == "boolean":
        normalized = []
        for value in values:
            if isinstance(value, bool):
                normalized.append(value)
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    normalized.append(True)
                elif lowered in {"false", "0", "no", "n"}:
                    normalized.append(False)
        if not normalized:
            return None
        return bool(pd.Series(normalized).mode(dropna=True).iloc[0])
    mode = pd.Series([str(value) for value in values]).mode(dropna=True)
    return clean_json(mode.iloc[0] if not mode.empty else values[0])


def build_prediction_input_schema(trained_content: dict[str, Any], split_artifact: dict[str, Any] | None):
    split_content = (split_artifact or {}).get("content") or {}
    splits = split_content.get("splits") or {}
    sample_rows = (splits.get("validation") or []) + (splits.get("train") or [])
    fields = []
    for field in usable_prediction_columns(trained_content):
        name = field["name"]
        field_type = field["type"]
        fields.append({
            "name": name,
            "label": name,
            "type": field_type,
            "required": False,
            "example": example_for_column(sample_rows, name, field_type),
            "helper_text": {
                "number": "Numeric input used by the model.",
                "text": "Categorical input used by the model.",
                "boolean": "Boolean input used by the model.",
            }[field_type],
        })
    return {"fields": fields}


def split_counts_and_ratios(split_artifact: dict[str, Any] | None):
    split_content = (split_artifact or {}).get("content") or {}
    splits = split_content.get("splits") or {}
    train_count = len(splits.get("train") or [])
    validation_count = len(splits.get("validation") or [])
    test_count = len(splits.get("test") or [])
    total = train_count + validation_count + test_count
    counts = {"training": train_count, "validation": validation_count, "test": test_count, "total": total}
    ratios = None
    if total > 0:
        ratios = {"training": train_count / total, "validation": validation_count / total, "test": test_count / total}
    return counts, ratios


def sample_validation_input(trained_content: dict[str, Any], split_artifact: dict[str, Any] | None, exclude_indices: list[int] | None = None):
    target_column = trained_content.get("target_column")
    split_content = (split_artifact or {}).get("content") or {}
    splits = split_content.get("splits") or {}
    validation_rows = [row for row in (splits.get("validation") or []) if isinstance(row, dict)]
    if not validation_rows:
        raise HTTPException(status_code=400, detail="No validation rows are available for sample prediction.")
    fields = usable_prediction_columns(trained_content)
    if not fields:
        raise HTTPException(status_code=400, detail="No usable input fields were found for this trained model.")
    candidates = [(index, row) for index, row in enumerate(validation_rows) if any(field["name"] in row for field in fields)]
    if not candidates:
        raise HTTPException(status_code=400, detail="No validation rows contain usable input fields for sample prediction.")
    excluded = {index for index in (exclude_indices or []) if isinstance(index, int)}
    available = [(index, row) for index, row in candidates if index not in excluded]
    sample_index, sample = random.choice(available or candidates)
    input_row = {field["name"]: clean_json(sample.get(field["name"])) for field in fields}
    counts, ratios = split_counts_and_ratios(split_artifact)
    sample_context = {
        "kind": "validation_row",
        "validation_row_index": sample_index,
        "validation_row_number": sample_index + 1,
        "validation_rows_total": len(validation_rows),
        "target_is_available_after_prediction": bool(target_column and target_column in sample),
        "split_counts": counts,
        "split_ratios": clean_json(ratios),
    }
    return input_row, sample_context


def normalize_prediction_value(value: Any, field_type: str):
    if value == "" or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if field_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if field_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)
    return str(value).strip().lower()


def prediction_values_match(left: Any, right: Any, field_type: str):
    normalized_left = normalize_prediction_value(left, field_type)
    normalized_right = normalize_prediction_value(right, field_type)
    if normalized_left is None or normalized_right is None:
        return normalized_left is None and normalized_right is None
    if field_type == "number":
        return bool(np.isclose(normalized_left, normalized_right, equal_nan=True))
    return normalized_left == normalized_right


def validation_row_for_context(split_artifact: dict[str, Any] | None, sample_context: dict[str, Any] | None):
    if not sample_context or sample_context.get("kind") != "validation_row":
        return None, None
    index = sample_context.get("validation_row_index")
    if not isinstance(index, int):
        return None, None
    split_content = (split_artifact or {}).get("content") or {}
    validation_rows = [row for row in ((split_content.get("splits") or {}).get("validation") or []) if isinstance(row, dict)]
    if index < 0 or index >= len(validation_rows):
        return None, None
    return index, validation_rows[index]


def build_prediction_provenance_and_ground_truth(
    task_type: str,
    target_column: str,
    prediction_value: Any,
    row: dict[str, Any],
    fields: list[dict[str, Any]],
    split_artifact: dict[str, Any] | None,
    sample_context: dict[str, Any] | None,
):
    counts, ratios = split_counts_and_ratios(split_artifact)
    validation_index, validation_row = validation_row_for_context(split_artifact, sample_context)
    validation_rows_total = counts.get("validation") if counts else None
    matched_validation_row = False
    if validation_row is not None:
        matched_validation_row = all(prediction_values_match(row.get(field["name"]), validation_row.get(field["name"]), field["type"]) for field in fields)
    if matched_validation_row and validation_row is not None and target_column in validation_row:
        actual_value = clean_json(validation_row.get(target_column))
        matches_prediction = None
        absolute_error = None
        if task_type == "tabular_classification":
            matches_prediction = prediction_values_match(prediction_value, actual_value, "text")
        else:
            predicted_number = normalize_prediction_value(prediction_value, "number")
            actual_number = normalize_prediction_value(actual_value, "number")
            if predicted_number is not None and actual_number is not None:
                absolute_error = abs(predicted_number - actual_number)
        provenance = {
            "kind": "validation_row",
            "validation_row_number": validation_index + 1 if validation_index is not None else None,
            "validation_rows_total": validation_rows_total,
            "split_counts": counts,
            "split_ratios": clean_json(ratios),
            "message": "This example comes from the validation split and was not used to train the model.",
        }
        ground_truth = {
            "available": True,
            "target_column": target_column,
            "actual_value": actual_value,
            "matches_prediction": matches_prediction,
            "absolute_error": clean_json(absolute_error),
        }
        return provenance, ground_truth
    message = "You changed the sampled values, so this is now a custom input. There is no known answer to compare against." if sample_context else "This is a custom input. There is no known answer to compare against."
    provenance = {
        "kind": "custom_input",
        "validation_row_number": None,
        "validation_rows_total": validation_rows_total,
        "split_counts": counts if counts.get("total") else None,
        "split_ratios": clean_json(ratios),
        "message": message,
    }
    ground_truth = {
        "available": False,
        "target_column": target_column,
        "actual_value": None,
        "matches_prediction": None,
        "absolute_error": None,
    }
    return provenance, ground_truth


def coerce_prediction_input(raw_input: dict[str, Any], fields: list[dict[str, Any]]):
    if not isinstance(raw_input, dict):
        raise HTTPException(status_code=400, detail="Prediction input must be a JSON object.")
    row = {}
    for field in fields:
        name = field["name"]
        value = raw_input.get(name)
        if value == "":
            value = None
        if value is None:
            row[name] = None
            continue
        if field["type"] == "number":
            try:
                row[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"{name} must be a number.") from exc
        elif field["type"] == "boolean":
            if isinstance(value, bool):
                row[name] = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    row[name] = True
                elif lowered in {"false", "0", "no", "n", "off"}:
                    row[name] = False
                else:
                    raise HTTPException(status_code=400, detail=f"{name} must be true or false.")
            else:
                row[name] = bool(value)
        else:
            row[name] = str(value)
    return row


def class_probabilities_for_pipeline(pipeline, frame: pd.DataFrame):
    if not hasattr(pipeline, "predict_proba"):
        return None, None
    probabilities = pipeline.predict_proba(frame)[0]
    model = getattr(pipeline, "named_steps", {}).get("model")
    classes = [clean_json(item) for item in getattr(model, "classes_", [])]
    if not classes:
        return None, None
    class_probabilities = {str(label): clean_json(prob) for label, prob in zip(classes, probabilities)}
    confidence = float(np.max(probabilities)) if len(probabilities) else None
    return class_probabilities, confidence


def build_plain_prediction_result(task_type: str, prediction_value: Any, confidence: float | None):
    if task_type == "tabular_classification":
        if confidence is not None:
            return f"For this input, the model predicts {prediction_value} with {confidence * 100:.0f}% confidence."
        return f"For this input, the model predicts {prediction_value}."
    return f"For this input, the model predicts a value of {prediction_value}."


@app.post("/test-prediction-schema")
def prediction_schema(request: PredictionSchemaRequest, authorization: str | None = Header(default=None)):
    require_config()
    user = require_user(authorization)
    try:
        load_owned_pipe(request.pipe_id, user["id"])
        _, trained_artifact_id, trained_content, split_artifact = load_prediction_lineage(request.pipe_id, request.review_results_artifact_id)
        schema = build_prediction_input_schema(trained_content, split_artifact)
        return {
            "previous_trained_models_artifact_id": trained_artifact_id,
            "task_type": trained_content.get("task_type"),
            "target_column": trained_content.get("target_column"),
            "model": {
                "model_id": trained_content.get("recommended_model_id"),
                "model_name": trained_content.get("recommended_model_name"),
            },
            "input_schema": schema,
        }
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/test-prediction-sample")
def prediction_sample(request: TestPredictionSampleRequest, authorization: str | None = Header(default=None)):
    require_config()
    user = require_user(authorization)
    try:
        load_owned_pipe(request.pipe_id, user["id"])
        _, _, trained_content, split_artifact = load_prediction_lineage(request.pipe_id, request.review_results_artifact_id)
        input_row, sample_context = sample_validation_input(trained_content, split_artifact, request.exclude_validation_row_indices)
        return {
            "input": input_row,
            "sample_context": sample_context,
            "source": {"kind": "validation_row", "description": "Real held-out validation row"},
        }
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/test-prediction")
def test_prediction(request: TestPredictionRequest, authorization: str | None = Header(default=None)):
    require_config()
    user = require_user(authorization)
    try:
        load_owned_pipe(request.pipe_id, user["id"])
        _, trained_artifact_id, trained_content, split_artifact = load_prediction_lineage(request.pipe_id, request.review_results_artifact_id)
        task_type = trained_content.get("task_type")
        target_column = trained_content.get("target_column")
        if task_type not in {"tabular_classification", "tabular_regression"} or not target_column:
            raise HTTPException(status_code=400, detail="Unsupported or missing task metadata.")
        schema = build_prediction_input_schema(trained_content, split_artifact)
        fields = schema.get("fields") or []
        if not fields:
            raise HTTPException(status_code=400, detail="No usable input fields were found for this trained model.")
        row = coerce_prediction_input(request.input, fields)
        frame = pd.DataFrame([row], columns=[field["name"] for field in fields])
        pipeline = decode_recommended_pipeline(trained_content.get("recommended_model_bundle") or {})
        prediction_raw = pipeline.predict(frame)[0]
        prediction_value = clean_json(prediction_raw)
        class_probabilities, confidence = (None, None)
        if task_type == "tabular_classification":
            class_probabilities, confidence = class_probabilities_for_pipeline(pipeline, frame)
        model = {
            "model_id": trained_content.get("recommended_model_id"),
            "model_name": trained_content.get("recommended_model_name"),
        }
        prediction = {
            "value": prediction_value,
            "label": str(prediction_value),
            "confidence": clean_json(confidence),
            "class_probabilities": clean_json(class_probabilities),
        }
        provenance, ground_truth = build_prediction_provenance_and_ground_truth(task_type, target_column, prediction_value, row, fields, split_artifact, request.sample_context)
        mappable_output = {
            "prediction": prediction_value,
            "confidence": clean_json(confidence),
            "class_probabilities": clean_json(class_probabilities),
            "model_name": model.get("model_name"),
            "pipe_id": request.pipe_id,
            "pipe_version": "draft",
        }
        plain_result = build_plain_prediction_result(task_type, prediction_value, confidence)
        content = clean_json({
            "previous_review_results_artifact_id": request.review_results_artifact_id,
            "previous_trained_models_artifact_id": trained_artifact_id,
            "task_type": task_type,
            "target_column": target_column,
            "model": model,
            "input_schema": schema,
            "input": row,
            "prediction": prediction,
            "plain_english_result": plain_result,
            "provenance": provenance,
            "ground_truth": ground_truth,
            "mappable_output": mappable_output,
        })
        artifact_payload = {
            "pipe_id": request.pipe_id,
            "artifact_type": "test_prediction",
            "kind": "test_prediction",
            "name": "Test prediction",
            "content": content,
            "metadata": {
                "previous_review_results_artifact_id": request.review_results_artifact_id,
                "previous_trained_models_artifact_id": trained_artifact_id,
                "task_type": task_type,
                "target_column": target_column,
                "model_name": model.get("model_name"),
                "prediction": prediction_value,
                "confidence": clean_json(confidence),
                "provenance_kind": provenance.get("kind"),
                "actual_value": ground_truth.get("actual_value"),
                "matches_prediction": ground_truth.get("matches_prediction"),
                "absolute_error": ground_truth.get("absolute_error"),
            },
        }
        artifact = rest_post("artifacts", artifact_payload)[0]
        output = {
            "step_key": "test_prediction",
            "status": "completed",
            "test_prediction_artifact_id": artifact["id"],
            "previous_review_results_artifact_id": request.review_results_artifact_id,
            "prediction": prediction_value,
            "confidence": clean_json(confidence),
            "model_name": model.get("model_name"),
            "provenance": provenance,
            "ground_truth": ground_truth,
            "storage": {"format": "json", "uri": f"artifact:{artifact['id']}"},
        }
        rest_post(
            "pipe_step_outputs",
            {"pipe_id": request.pipe_id, "step_key": "test_prediction", "artifact_id": artifact["id"], "status": "completed", "output": output},
            params={"on_conflict": "pipe_id,step_key"},
            prefer="resolution=merge-duplicates,return=representation",
        )
        return {"test_prediction_artifact_id": artifact["id"], **content}
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {exc.response.text[:500]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
