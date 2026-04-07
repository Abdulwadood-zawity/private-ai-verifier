from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel


class AttestationReport(BaseModel):
    provider: str  # e.g., "tinfoil", "redpill", "nearai", "chutes"
    model_id: Optional[str] = None
    intel_quote: str  # Hex string
    request_nonce: Optional[str] = None
    # Dict for NRAS / Phala-style payloads (`{nonce, arch, evidence_list:[…]}`),
    # List for Chutes and the flat NEAR shape ([{cert, evidence, arch}, …]).
    nvidia_payload: Optional[Union[Dict[str, Any], List[Any]]] = None
    raw: Optional[Any] = None


# Hardware Types
HARDWARE_INTEL_TDX = "INTEL_TDX"
HARDWARE_AMD_SEV_SNP = "AMD_SEV_SNP"
HARDWARE_NVIDIA_CC = "NVIDIA_CC"


class GpuEvidence(BaseModel):
    """Single GPU's certificate + evidence pair from the upstream nvidia_payload."""

    arch: Optional[str] = None
    certificate: Optional[str] = None
    evidence: Optional[str] = None


class MessageSignature(BaseModel):
    """NEAR-style per-message ECDSA signature.

    Mirrors the shape used by the o.llm Python proxy so the gateway can save
    these fields verbatim into the `attestations` table.
    """

    ecdsa_signature: Optional[str] = None
    message_signer: Optional[str] = None
    model_signing_address: Optional[str] = None
    request_hash: Optional[str] = None
    response_hash: Optional[str] = None
    text: Optional[str] = None  # full "request_hash:response_hash" string
    signing_algo: Optional[str] = None


class VerificationResult(BaseModel):
    model_verified: bool
    provider: str
    timestamp: float
    hardware_type: List[str]  # e.g., ["INTEL_TDX", "NVIDIA_CC"]
    model_id: Optional[str] = None
    request_nonce: Optional[str] = None
    signing_address: Optional[str] = None
    claims: Dict[str, Any]
    error: Optional[str] = None
    raw: Optional[Any] = None
    # Raw Intel TDX quote (hex / base64) carried through from the upstream
    # provider so consumers (e.g. the o.llm gateway) can persist it verbatim.
    intel_quote: Optional[str] = None

    # ─── Extra fields for parity with the o.llm Python attestation pipeline ───
    # Raw upstream nvidia_payload broken into per-GPU rows. Gateway uses this
    # to populate `attestation_gpu_evidence` with one row per ordinal.
    nvidia_evidence_list: Optional[List[GpuEvidence]] = None
    # Nonce reported inside the upstream nvidia_payload (used by spend_logs).
    nvidia_payload_nonce: Optional[str] = None
    # Top-level architecture string from the upstream nvidia_payload.
    nvidia_arch: Optional[str] = None
    # ISO 8601 timestamp string for `attestations.attestation_timestamp`.
    attestation_timestamp_iso: Optional[str] = None
    # NEAR-only: per-message ECDSA signature pulled from /v1/signature/{chat_id}.
    message_signature: Optional[MessageSignature] = None
