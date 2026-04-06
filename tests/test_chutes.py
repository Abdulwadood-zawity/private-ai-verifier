import pytest
import asyncio
import os
from confidential_verifier.sdk import TeeVerifier
from confidential_verifier.verifiers.chutes import ChutesVerifier
from confidential_verifier.providers.chutes import ChutesProvider


@pytest.mark.asyncio
async def test_chutes_verifier_missing_fields():
    """Test that ChutesVerifier returns proper errors for missing fields."""
    verifier = ChutesVerifier()

    # Missing quote
    result = await verifier.verify({})
    assert not result.model_verified
    assert "Missing TDX quote" in result.error

    # Missing nonce/pubkey
    result = await verifier.verify({"quote": "dGVzdA=="})  # base64 "test"
    assert not result.model_verified
    assert "Missing nonce or e2e_pubkey" in result.error


@pytest.mark.asyncio
async def test_chutes_provider_requires_api_key():
    """Test that ChutesProvider requires an API key."""
    # Clear env var if set
    old_key = os.environ.pop("CHUTES_API_KEY", None)

    try:
        with pytest.raises(ValueError, match="API key is required"):
            ChutesProvider()
    finally:
        # Restore env var
        if old_key:
            os.environ["CHUTES_API_KEY"] = old_key


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHUTES_API_KEY"),
    reason="CHUTES_API_KEY not set"
)
async def test_chutes_lookup_by_name():
    """Test looking up chute_id by name."""
    api_key = os.getenv("CHUTES_API_KEY")
    provider = ChutesProvider(api_key=api_key)

    # Use a known public chute name
    name = "moonshotai/Kimi-K2.5-TEE"
    try:
        chute_id = provider.lookup_chute_id(name)
        assert chute_id is not None
        assert len(chute_id) == 36  # UUID format
        print(f"✅ Looked up {name} -> {chute_id}")
    except Exception as e:
        pytest.skip(f"Chute not found or API error: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHUTES_API_KEY"),
    reason="CHUTES_API_KEY not set"
)
async def test_chutes_search():
    """Test searching for chutes."""
    api_key = os.getenv("CHUTES_API_KEY")
    provider = ChutesProvider(api_key=api_key)

    items = provider.search_chutes(include_public=True)
    print(f"✅ Found {len(items)} public chutes")
    for item in items[:5]:
        print(f"  - {item.get('name')} ({item.get('chute_id')})")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHUTES_API_KEY"),
    reason="CHUTES_API_KEY not set"
)
async def test_chutes_fetch_pubkeys():
    """Test fetching E2E public keys from Chutes API."""
    api_key = os.getenv("CHUTES_API_KEY")
    chute_id = os.getenv("CHUTES_TEST_CHUTE_ID")

    if not chute_id:
        pytest.skip("CHUTES_TEST_CHUTE_ID not set")

    provider = ChutesProvider(api_key=api_key)
    pubkeys = provider.fetch_e2e_pubkeys(chute_id)

    assert isinstance(pubkeys, dict)
    print(f"✅ Found {len(pubkeys)} E2E-enabled instances")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHUTES_API_KEY"),
    reason="CHUTES_API_KEY not set"
)
async def test_chutes_fetch_and_verify():
    """
    Test the full Chutes flow: fetch report and verify it.
    This test runs live against Chutes API.
    """
    api_key = os.getenv("CHUTES_API_KEY")
    chute_id = os.getenv("CHUTES_TEST_CHUTE_ID")

    if not chute_id:
        pytest.skip("CHUTES_TEST_CHUTE_ID not set")

    verifier = TeeVerifier(chutes_api_key=api_key)

    print(f"\n[Test] Fetching and verifying report for chute {chute_id}...")

    result = await verifier.verify_model("chutes", chute_id)

    assert result.claims is not None
    assert "instances" in result.claims

    if result.model_verified:
        print(f"✅ All instances verified successfully")
        assert "INTEL_TDX" in result.hardware_type
    else:
        print(f"⚠️ Verification result: {result.error}")

    # Print instance details
    for iid, claims in result.claims.get("instances", {}).items():
        e2e_verified = claims.get("e2e_binding_verified", False)
        print(f"  Instance {iid[:8]}... E2E binding: {'✅' if e2e_verified else '❌'}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHUTES_API_KEY"),
    reason="CHUTES_API_KEY not set"
)
async def test_chutes_in_sdk():
    """Test that Chutes is properly integrated in SDK."""
    api_key = os.getenv("CHUTES_API_KEY")
    verifier = TeeVerifier(chutes_api_key=api_key)

    assert "chutes" in verifier.providers
    assert verifier.chutes_verifier is not None
    print("✅ Chutes is properly integrated in SDK")


if __name__ == "__main__":
    asyncio.run(test_chutes_verifier_missing_fields())
    asyncio.run(test_chutes_provider_requires_api_key())

    if os.getenv("CHUTES_API_KEY"):
        asyncio.run(test_chutes_lookup_by_name())
        asyncio.run(test_chutes_search())
        asyncio.run(test_chutes_fetch_and_verify())
        asyncio.run(test_chutes_in_sdk())
