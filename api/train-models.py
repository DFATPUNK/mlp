import base64
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def response(handler, status, payload):
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error(handler, status, message):
    response(handler, status, {"error": message})


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


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.split("?")[0] not in {"/api/train-models", "/api/train-models.py"}:
            return error(self, 404, "Not found")
        if not SUPABASE_URL or not SERVICE_ROLE_KEY:
            return error(self, 500, "Training API is missing server configuration.")

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return error(self, 401, "Missing Authorization bearer token.")
        user = verify_user(auth_header.replace("Bearer ", "", 1).strip())
        if not user or not user.get("id"):
            return error(self, 401, "Invalid Supabase session.")

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            pipe_id = payload.get("pipe_id")
            target_config_artifact_id = payload.get("target_config_artifact_id")
            if not pipe_id or not target_config_artifact_id:
                return error(self, 400, "pipe_id and target_config_artifact_id are required.")

            pipe = get_single("pipes", {"id": f"eq.{pipe_id}", "select": "id,owner_id"})
            if not pipe or pipe.get("owner_id") != user["id"]:
                return error(self, 403, "You do not have access to train this pipe.")

            target_artifact = get_single("artifacts", {"id": f"eq.{target_config_artifact_id}", "pipe_id": f"eq.{pipe_id}", "select": "id,content"})
            if not target_artifact:
                return error(self, 404, "Target configuration artifact was not found.")
            target_config = target_artifact.get("content") or {}
            split_artifact_id = target_config.get("previous_split_dataset_artifact_id")
            split_artifact = get_single("artifacts", {"id": f"eq.{split_artifact_id}", "pipe_id": f"eq.{pipe_id}", "select": "id,content"})
            if not split_artifact:
                return error(self, 404, "Split dataset artifact was not found.")

            split_content = split_artifact.get("content") or {}
            train_rows = (split_content.get("splits") or {}).get("train") or []
            validation_rows = (split_content.get("splits") or {}).get("validation") or []
            target_column = target_config.get("target_column")
            task_type = target_config.get("detected_task_type")
            feature_columns = target_config.get("feature_columns") or []
            excluded_columns = set(target_config.get("excluded_feature_columns") or [])
            if task_type not in {"tabular_classification", "tabular_regression"}:
                return error(self, 400, "Unsupported task type.")
            if not train_rows or not validation_rows:
                return error(self, 400, "Train and validation splits are required.")

            train_df, dropped_train = rows_to_frame(train_rows, target_column)
            val_df, dropped_val = rows_to_frame(validation_rows, target_column)
            if train_df.empty or val_df.empty:
                return error(self, 400, "Not enough rows remain after dropping missing targets.")

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
                "previous_target_config_artifact_id": target_config_artifact_id,
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
                "pipe_id": pipe_id,
                "artifact_type": "trained_models",
                "kind": "trained_models",
                "name": "Trained models",
                "content": content,
                "metadata": {
                    "previous_target_config_artifact_id": target_config_artifact_id,
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
                "previous_target_config_artifact_id": target_config_artifact_id,
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
                {"pipe_id": pipe_id, "step_key": "train_models", "artifact_id": artifact["id"], "status": "completed", "output": output},
                params={"on_conflict": "pipe_id,step_key"},
                prefer="resolution=merge-duplicates,return=representation",
            )
            response(self, 200, {
                "trained_models_artifact_id": artifact["id"],
                "recommended_model_id": recommended["model_id"],
                "recommended_model_name": recommended["model_name"],
                "primary_metric_name": recommended["primary_metric_name"],
                "primary_metric_value": recommended["primary_metric_value"],
                "model_count": output["model_count"],
                "task_type": task_type,
                "target_column": target_column,
            })
        except requests.HTTPError as exc:
            error(self, 500, f"Supabase request failed: {exc.response.text[:500]}")
        except Exception as exc:
            error(self, 500, str(exc))
