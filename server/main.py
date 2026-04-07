import logging
import traceback

from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from confidential_verifier.sdk import TeeVerifier
from confidential_verifier.types import AttestationReport, VerificationResult

logger = logging.getLogger("verifier")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Confidential Service Verifier API")
verifier = TeeVerifier()


@app.get("/providers")
def list_providers():
    return verifier.list_providers()


@app.get("/models")
async def list_models(provider: str):
    try:
        models = await verifier.list_models(provider)
        return models
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch-report")
async def fetch_report(provider: str, model_id: str):
    try:
        report = await verifier.fetch_report(provider, model_id)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_report(report: AttestationReport):
    try:
        result = await verifier.verify(report)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/verify-model")
async def verify_model(
    provider: str,
    model_id: str,
    chat_id: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream chat completion id. When provided with "
            "provider=nearai, the verifier additionally fetches the per-message "
            "ECDSA signature from cloud-api.near.ai/v1/signature/{chat_id} and "
            "attaches it as `message_signature` on the result."
        ),
    ),
):
    try:
        result = await verifier.verify_model(provider, model_id, chat_id=chat_id)
        return result
    except ValueError as e:
        logger.error("verify_model 400 (provider=%s model=%s): %s", provider, model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the full traceback so the actual error is visible in stdout
        # instead of being swallowed by the catch-all and returned as an
        # opaque 500.
        logger.error(
            "verify_model 500 (provider=%s model=%s chat_id=%s): %s\n%s",
            provider,
            model_id,
            chat_id,
            e,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
