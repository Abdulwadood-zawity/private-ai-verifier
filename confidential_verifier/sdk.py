import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional
from .types import (
    AttestationReport,
    GpuEvidence,
    MessageSignature,
    VerificationResult,
)
from .providers import (
    TinfoilProvider,
    RedpillProvider,
    NearaiProvider,
    ChutesProvider,
)
from .verifiers import (
    NvidiaGpuVerifier,
    NearAICloudVerifier,
    RedpillVerifier,
    ChutesVerifier,
)


class TeeVerifier:
    def __init__(self, chutes_api_key: Optional[str] = None):
        self.providers = {
            "tinfoil": TinfoilProvider(),
            "redpill": RedpillProvider(),
            "nearai": NearaiProvider(),
        }
        # Chutes requires API key, only add if available
        chutes_key = chutes_api_key or os.getenv("CHUTES_API_KEY")
        if chutes_key:
            self.providers["chutes"] = ChutesProvider(api_key=chutes_key)

        self.nvidia_verifier = NvidiaGpuVerifier()
        self.nearai_verifier = NearAICloudVerifier()
        self.redpill_verifier = RedpillVerifier()
        self.chutes_verifier = ChutesVerifier()

    async def fetch_report(
        self, provider_name: str, model_id: str
    ) -> AttestationReport:
        provider = self.providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        # Run sync provider.fetch_report in thread pool to avoid blocking event loop
        return await asyncio.to_thread(provider.fetch_report, model_id)

    async def verify(self, report: AttestationReport) -> VerificationResult:
        # Get provider from report
        provider_name = report.provider.lower()

        # Special handling for NearAI which has a complex multi-component report
        if provider_name == "nearai":
            if not report.raw:
                return VerificationResult(
                    model_verified=False,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=["INTEL_TDX"],  # fallback
                    claims={},
                    error="Missing raw report data for NearAI verification",
                )
            return await self.nearai_verifier.verify(
                report.raw,
                request_nonce=report.request_nonce,
                model_id=report.model_id,
            )

        # Special handling for Redpill which uses PhalaCloudVerifier internally
        if provider_name == "redpill":
            if not report.raw:
                return VerificationResult(
                    model_verified=False,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=["INTEL_TDX"],
                    claims={},
                    error="Missing raw report data for Redpill verification",
                )
            # Build report data with all required fields
            report_data = {
                **report.raw,
                "request_nonce": report.request_nonce,
                "nvidia_payload": report.nvidia_payload,
            }
            return await self.redpill_verifier.verify(report_data)

        # Special handling for Chutes which needs E2E pubkey binding verification
        if provider_name == "chutes":
            if not report.raw:
                return VerificationResult(
                    model_verified=False,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=[],
                    claims={},
                    error="Missing raw report data for Chutes verification",
                )
            # Verify all instances
            nonce = report.raw.get("nonce") or report.request_nonce
            pubkeys = report.raw.get("pubkeys", {})
            instances_evidence = report.raw.get("evidence", [])

            if not instances_evidence:
                return VerificationResult(
                    model_verified=False,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=[],
                    claims={},
                    error="No instance evidence found in Chutes report",
                )

            # Verify all instances and aggregate results
            results = await self.chutes_verifier.verify_multiple_instances(
                instances_evidence, nonce, pubkeys
            )

            # all([]) returns True, so we must check for empty results
            if not results:
                return VerificationResult(
                    model_verified=False,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=[],
                    model_id=report.model_id,
                    request_nonce=nonce,
                    claims={},
                    error="No valid instances found to verify",
                )

            all_verified = all(r.model_verified for r in results.values())
            combined_claims = {
                "instances": {iid: r.claims for iid, r in results.items()},
                "chute_id": report.model_id,
            }
            errors = [
                f"{iid}: {r.error}" for iid, r in results.items() if r.error
            ]

            hardware_types = set()
            for r in results.values():
                hardware_types.update(r.hardware_type)

            return VerificationResult(
                model_verified=all_verified,
                provider=provider_name,
                timestamp=time.time(),
                hardware_type=list(hardware_types),
                model_id=report.model_id,
                request_nonce=nonce,
                claims=combined_claims,
                error="; ".join(errors) if errors else None,
            )

        provider = self.providers.get(provider_name)
        if not provider:
            # Fallback for reports that might have been saved before this change
            # or from other sources. Use a default IntelTdxVerifier which does
            # trivial verification (no policy).
            from .verifiers import IntelTdxVerifier

            intel_verifier = IntelTdxVerifier()
        else:
            intel_verifier = provider.get_verifier()

        # Wrap quote with metadata if available in raw
        quote_input = report.intel_quote
        if isinstance(report.raw, dict):
            quote_input = {
                "quote": report.intel_quote,
                "model_id": report.raw.get("model_id"),
                "repo": report.raw.get("repo"),
                "request_nonce": report.request_nonce,
                "signing_address": report.raw.get("signing_address"),
                # Tinfoil-specific fields for format detection
                "quote_type": report.raw.get("quote_type"),
                "format": report.raw.get("format"),
            }

        # 1. Verify Intel TDX Quote (Mandatory)
        intel_result = await intel_verifier.verify(quote_input)

        if not intel_result.model_verified:
            return intel_result

        # 2. Verify Nvidia CC Payload if present
        if report.nvidia_payload:
            nvidia_result = await self.nvidia_verifier.verify(report.nvidia_payload)

            # Combine claims
            combined_claims = {
                "intel": intel_result.claims,
                "nvidia": nvidia_result.claims,
            }

            if nvidia_result.model_verified:
                return VerificationResult(
                    model_verified=True,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=["INTEL_TDX", "NVIDIA_CC"],
                    claims=combined_claims,
                    raw={"intel": intel_result.raw, "nvidia": nvidia_result.raw},
                )
            else:
                return VerificationResult(
                    model_verified=intel_result.model_verified,
                    provider=provider_name,
                    timestamp=time.time(),
                    hardware_type=["INTEL_TDX", "NVIDIA_CC"],
                    claims=combined_claims,
                    raw={"intel": intel_result.raw, "nvidia": nvidia_result.raw},
                    error=nvidia_result.error,
                )

        return intel_result

    async def verify_model(
        self,
        provider_name: str,
        model_id: str,
        chat_id: Optional[str] = None,
    ) -> VerificationResult:
        """Fetch a report from a provider and verify it.

        The raw Intel TDX quote from the provider is carried through to the
        VerificationResult so consumers can persist it.

        If `chat_id` is provided AND the provider is `nearai`, also fetches the
        per-message ECDSA signature from cloud-api.near.ai/v1/signature/{chat_id}
        and attaches it to the result so the gateway can persist it alongside
        the rest of the attestation.
        """
        report = await self.fetch_report(provider_name, model_id)
        result = await self.verify(report)
        # Always carry the raw upstream quote through (existing upstream behaviour).
        result.intel_quote = report.intel_quote
        # Echo the raw upstream nvidia_payload and incoming chat_id so the
        # /verify-model response shape matches the o.llm Python attestation
        # callback. Downstream consumers (Rust gateway, UI) read these top-level
        # fields directly.
        result.nvidia_payload = report.nvidia_payload
        result.chat_id = chat_id
        # Project per-GPU rows / nonce / arch / iso timestamp onto the result.
        _enrich_with_raw_fields(result, report)

        if chat_id and provider_name.lower() == "nearai":
            try:
                sig = await _fetch_near_message_signature(model_id, chat_id)
            except Exception:
                # Per-message signature is best-effort: a failure here must not
                # break the model attestation, which is the primary payload.
                sig = None
            if sig is not None:
                # The model signing address is set elsewhere on the result; the
                # gateway falls back to it if this nested field is empty.
                if not sig.model_signing_address and result.signing_address:
                    sig.model_signing_address = result.signing_address
                result.message_signature = sig

        return result

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    async def list_models(self, provider_name: str) -> List[str]:
        provider = self.providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider.list_models()


# ─────────────────────── helpers ───────────────────────


def _enrich_with_raw_fields(
    result: VerificationResult, report: AttestationReport
) -> None:
    """Copy raw upstream fields from the AttestationReport into the result.

    The verifier itself only emits processed JWT claims, but the o.llm gateway
    needs the original per-GPU certificate/evidence pairs (so it can write one
    row per GPU to `attestation_gpu_evidence`) plus the upstream nonce, arch,
    and an ISO timestamp for `attestations.attestation_timestamp`.
    """
    result.attestation_timestamp_iso = datetime.now(timezone.utc).isoformat()

    nvidia_payload: Any = report.nvidia_payload
    if isinstance(nvidia_payload, str):
        try:
            import json

            nvidia_payload = json.loads(nvidia_payload)
        except Exception:
            nvidia_payload = None

    evidence_list: List[GpuEvidence] = []
    arch: Optional[str] = None
    nonce: Optional[str] = None

    if isinstance(nvidia_payload, dict):
        nonce = nvidia_payload.get("nonce")
        arch = nvidia_payload.get("arch")
        raw_list = nvidia_payload.get("evidence_list")
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict):
                    evidence_list.append(
                        GpuEvidence(
                            arch=item.get("arch") or arch,
                            certificate=item.get("certificate"),
                            evidence=item.get("evidence"),
                        )
                    )
    elif isinstance(nvidia_payload, list):
        # NEAR / Chutes sometimes return nvidia_payload as a flat list of
        # evidence dicts. Each item carries its own arch, certificate, evidence.
        for item in nvidia_payload:
            if isinstance(item, dict):
                if not arch and item.get("arch"):
                    arch = item.get("arch")
                evidence_list.append(
                    GpuEvidence(
                        arch=item.get("arch"),
                        certificate=item.get("certificate"),
                        evidence=item.get("evidence"),
                    )
                )

    if evidence_list:
        result.nvidia_evidence_list = evidence_list
    if nonce:
        result.nvidia_payload_nonce = nonce
    if arch:
        result.nvidia_arch = arch


async def _fetch_near_message_signature(
    model_id: str, chat_id: str, timeout: float = 10.0
) -> Optional[MessageSignature]:
    """Calls cloud-api.near.ai/v1/signature/{chat_id} and parses the response.

    Mirrors what `fetch_near_message_signature` in the o.llm Python proxy does:
    splits `text` on `:` to recover request/response hashes, captures
    `signature` and `signing_address`. Returns None on any failure (missing
    API key, network error, non-200 response, malformed body) so the caller
    can fall back to the model-attestation-only path without surfacing a 500.

    Uses the `requests` library (already a verifier dependency) wrapped in
    `asyncio.to_thread` so we don't pull in `httpx` and don't block the loop.
    """
    api_key = os.environ.get("NEAR_AI_API_KEY", "")
    if not api_key:
        return None

    # NEAR's signature API wants the bare model id (no near/ prefix).
    clean_model = model_id
    for prefix in ("near/", "nearai/", "near_ai/"):
        if clean_model.lower().startswith(prefix):
            clean_model = clean_model[len(prefix):]
            break

    url = (
        f"https://cloud-api.near.ai/v1/signature/{chat_id}"
        f"?model={clean_model}&signing_algo=ecdsa"
    )

    def _do_get() -> Optional[dict]:
        try:
            import requests  # local import keeps the module surface small
        except ImportError:
            return None
        try:
            resp = requests.get(
                url,
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                timeout=timeout,
            )
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    try:
        data = await asyncio.to_thread(_do_get)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    text = data.get("text", "") or ""
    request_hash: Optional[str] = None
    response_hash: Optional[str] = None
    if ":" in text:
        parts = text.split(":", 1)
        request_hash = parts[0]
        response_hash = parts[1] if len(parts) > 1 else None

    return MessageSignature(
        ecdsa_signature=data.get("signature"),
        message_signer=data.get("signing_address"),
        # The model signing address is filled in by `verify_model` from
        # `result.signing_address` when this nested field is empty.
        model_signing_address=None,
        request_hash=request_hash,
        response_hash=response_hash,
        text=text or None,
        signing_algo=data.get("signing_algo"),
    )
