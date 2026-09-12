"""main.py — Model serving API với FastAPI"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import numpy as np, time, logging, uuid
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_uri: str = "models:/churn-classifier@champion"
    threshold: float = 0.5
    max_batch: int = 64
    class Config:
        env_file = ".env"

settings = Settings()

# --- METRICS PROMETHEUS ---
REQUESTS = Counter("predict_requests_total", "Tổng request", ["status"])
LATENCY  = Histogram("predict_latency_seconds", "Độ trễ",
                     buckets=[.005,.01,.025,.05,.1,.25,.5,1,2.5])
SCORE    = Histogram("predict_score", "Phân bố điểm dự đoán",
                     buckets=np.arange(0, 1.05, 0.05).tolist())
MODEL_UP = Gauge("model_loaded", "Model đã nạp chưa")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nạp model MỘT LẦN khi khởi động, KHÔNG nạp trong mỗi request
    import mlflow.pyfunc
    log.info("Đang nạp model...")
    STATE["model"] = mlflow.pyfunc.load_model(settings.model_uri)
    MODEL_UP.set(1)
    log.info("Model đã sẵn sàng")
    yield
    STATE.clear(); MODEL_UP.set(0)


app = FastAPI(title="Churn API", version="1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class PredictRequest(BaseModel):
    features: list[list[float]] = Field(..., min_length=1, max_length=64)

class Prediction(BaseModel):
    probability: float
    label: int

class PredictResponse(BaseModel):
    request_id: str
    predictions: list[Prediction]
    latency_ms: float
    model_version: str


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/health")
async def health():
    """Liveness — process còn sống không."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness — đã sẵn sàng nhận traffic chưa. K8s dùng cái này."""
    if "model" not in STATE:
        raise HTTPException(503, "Model chưa nạp xong")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    try:
        X = np.asarray(req.features, dtype=np.float32)
        proba = STATE["model"].predict(X)
        proba = np.asarray(proba).reshape(-1)

        for p in proba:
            SCORE.observe(float(p))

        preds = [Prediction(probability=float(p),
                            label=int(p >= settings.threshold)) for p in proba]
        dt = time.perf_counter() - t0
        LATENCY.observe(dt)
        REQUESTS.labels(status="success").inc()

        log.info(f"rid={request.state.request_id} n={len(preds)} "
                 f"latency={dt*1000:.1f}ms")

        return PredictResponse(
            request_id=request.state.request_id,
            predictions=preds,
            latency_ms=round(dt * 1000, 2),
            model_version=settings.model_uri,
        )
    except Exception as e:
        REQUESTS.labels(status="error").inc()
        log.exception(f"rid={request.state.request_id}")
        raise HTTPException(500, f"Lỗi dự đoán: {e}")