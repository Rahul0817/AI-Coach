"""Unit tests for password hashing and JWT handling."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.exceptions import AuthenticationError
from app.core.security import (
    MAX_PASSWORD_BYTES,
    _create_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_reset_token,
    hash_password,
    hash_reset_token,
    token_expiry_seconds,
    verify_password,
)

pytestmark = pytest.mark.unit


class TestPasswordHashing:
    def test_hash_then_verify_roundtrip(self) -> None:
        hashed = hash_password("Oviora!Secure24")
        assert verify_password("Oviora!Secure24", hashed)

    def test_wrong_password_rejected(self) -> None:
        hashed = hash_password("Oviora!Secure24")
        assert not verify_password("Oviora!Secure25", hashed)

    def test_same_password_hashes_differently(self) -> None:
        """bcrypt salts each hash, so identical passwords must not collide.

        If this fails, the salt is not being applied and one leaked rainbow
        table would compromise every account sharing a password.
        """
        assert hash_password("Oviora!Secure24") != hash_password("Oviora!Secure24")

    def test_oversized_password_rejected(self) -> None:
        """bcrypt silently truncates past 72 bytes, so we reject explicitly.

        Without this, "<72 chars>" and "<72 chars>+anything" authenticate
        against the same hash.
        """
        with pytest.raises(ValueError):
            hash_password("a" * (MAX_PASSWORD_BYTES + 1))

    def test_oversized_candidate_fails_verification(self) -> None:
        hashed = hash_password("a" * 70)
        assert not verify_password("a" * 200, hashed)

    def test_malformed_hash_is_a_failed_login_not_a_crash(self) -> None:
        assert not verify_password("anything", "not-a-bcrypt-hash")


class TestTokens:
    def test_access_token_roundtrip(self) -> None:
        token = create_access_token("user-123")
        payload = decode_token(token, expected_type="access")
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"

    def test_refresh_token_cannot_be_used_as_access(self) -> None:
        """The `type` claim must be enforced.

        Without this check a long-lived refresh token would be accepted
        wherever a 30-minute access token is, defeating the point of having
        two token types.
        """
        token = create_refresh_token("user-123")
        with pytest.raises(AuthenticationError):
            decode_token(token, expected_type="access")

    def test_tampered_token_rejected(self) -> None:
        token = create_access_token("user-123")
        head, payload, signature = token.split(".")
        forged = f"{head}.{payload}.{signature[:-4]}AAAA"
        with pytest.raises(AuthenticationError):
            decode_token(forged)

    def test_expired_token_rejected(self) -> None:
        token = _create_token("user-123", "access", timedelta(seconds=-10))
        with pytest.raises(AuthenticationError):
            decode_token(token)

    def test_every_token_has_a_unique_jti(self) -> None:
        """The jti is what makes revocation possible; duplicates would let one
        logout invalidate an unrelated session."""
        first = decode_token(create_access_token("u"))["jti"]
        second = decode_token(create_access_token("u"))["jti"]
        assert first != second

    def test_extra_claims_are_carried(self) -> None:
        token = create_access_token("u", extra_claims={"email": "a@b.c"})
        assert decode_token(token)["email"] == "a@b.c"

    def test_expiry_seconds_never_negative(self) -> None:
        expired = _create_token("u", "access", timedelta(seconds=-60))
        # Read the claims without validating, since the token is expired by
        # design and decode_token would (correctly) reject it.
        import jose.jwt as jose_jwt

        from app.core.config import settings

        payload = jose_jwt.get_unverified_claims(expired)
        assert token_expiry_seconds(payload) == 0
        assert settings.algorithm == "HS256"

    def test_garbage_input_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            decode_token("not.a.token")


class TestResetTokens:
    def test_only_the_digest_is_storable(self) -> None:
        """A database leak must not hand an attacker working reset links."""
        raw, digest = generate_reset_token()
        assert raw != digest
        assert hash_reset_token(raw) == digest

    def test_digest_is_deterministic(self) -> None:
        raw, digest = generate_reset_token()
        assert hash_reset_token(raw) == hash_reset_token(raw) == digest
