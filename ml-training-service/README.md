# MLP Training Service

Cloud Run service for Step 5, **Train models**. The Vercel app calls this service instead of bundling the Python/scikit-learn stack in a Vercel Function.

## Local development

```bash
cd ml-training-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Required environment variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ALLOWED_ORIGINS` optional comma-separated list. For MVP, permissive CORS is acceptable because the service verifies the Supabase access token and pipe ownership before training.

The frontend must set:

```bash
VITE_ML_TRAINING_API_URL=https://YOUR_CLOUD_RUN_URL
```

## Deploy to Cloud Run

The intended Google Cloud project name is `ml_pipes`, the project ID is `ml-pipes-499509`, and the recommended region is `europe-west1`.

```bash
gcloud config set project ml-pipes-499509

gcloud run deploy mlp-training-service \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --timeout 300 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --set-env-vars SUPABASE_URL="https://YOUR_SUPABASE_PROJECT.supabase.co" \
  --set-secrets SUPABASE_SERVICE_ROLE_KEY=supabase-service-role-key:latest
```

## Cost safety notes

For MVP cost control:

- Keep `min-instances = 0`.
- Keep `max-instances = 1`.
- Do not use a GPU.
- Start with 2 CPU / 2Gi memory.
- Keep datasets small.
- Create Google Cloud budget alerts.

## API

### `GET /health`

Returns `{ "status": "ok" }`.

### `POST /train-models`

Headers:

```http
Authorization: Bearer <Supabase access token>
```

Body:

```json
{
  "pipe_id": "...",
  "target_config_artifact_id": "..."
}
```

The service verifies the Supabase token, checks that the pipe belongs to the authenticated user, trains real scikit-learn models, persists a `trained_models` artifact, upserts `pipe_step_outputs` with `step_key = "train_models"`, and returns a training summary.

## Review results endpoint

`POST /review-results` generates the Step 6 review from an existing `trained_models` artifact. It does **not** retrain models. The service verifies the Supabase access token, checks pipe ownership, loads the recommended model bundle plus validation split, recomputes predictions for the recommended model only, generates matplotlib charts, and persists:

- `artifacts.artifact_type = "review_results"`
- `pipe_step_outputs.step_key = "review_results"`

Request body:

```json
{
  "pipe_id": "PIPE_ID",
  "trained_models_artifact_id": "TRAINED_MODELS_ARTIFACT_ID"
}
```

The response includes the recommended model summary, model comparison, validation notes, base64 PNG charts, prediction examples, and `review_results_artifact_id` for Step 7.
