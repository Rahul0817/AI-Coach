"""Authentication request/response contracts.

The password policy lives here, in one validator, so registration and password
change can never drift apart.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, EmailStr, Field, field_validator

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 72  # bcrypt's hard limit — see core.security

_COMMON_PASSWORDS = frozenset(
    {
        "password12",
        "password123",
        "qwerty1234",
        "1234567890",
        "letmein123",
        "welcome123",
        "iloveyou12",
        "admin12345",
    }
)


def validate_password_strength(password: str) -> str:
    """Enforce length plus three of four character classes.

    Requiring *variety* rather than a rigid "one of each" rule keeps genuinely
    strong passphrases usable while still rejecting the obvious cases.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password is too long (maximum 72 bytes).")
    if password.lower() in _COMMON_PASSWORDS:
        raise ValueError("That password is too common. Please choose another.")

    classes = sum(
        bool(pattern.search(password))
        for pattern in (
            re.compile(r"[a-z]"),
            re.compile(r"[A-Z]"),
            re.compile(r"\d"),
            re.compile(r"[^A-Za-z0-9]"),
        )
    )
    if classes < 3:
        raise ValueError(
            "Password must combine at least three of: lowercase, uppercase, "
            "digits, symbols."
        )
    return password


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., examples=["asha@example.com"])
    password: str = Field(..., examples=["Oviora!Secure24"])
    full_name: str = Field(..., min_length=2, max_length=120, examples=["Asha Rao"])

    @field_validator("password")
    @classmethod
    def _strong(cls, v: str) -> str:
        return validate_password_strength(v)

    @field_validator("full_name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("Full name cannot be blank.")
        return cleaned


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=200)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 - the OAuth2 scheme name
    expires_in: int = Field(..., description="Access token lifetime in seconds.")


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=200)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _strong(cls, v: str) -> str:
        return validate_password_strength(v)


class DeleteAccountRequest(BaseModel):
    """Deleting an account is irreversible, so it requires re-authentication."""

    password: str = Field(..., min_length=1, max_length=200)
    confirmation: str = Field(
        ...,
        description='Must be exactly "DELETE MY ACCOUNT".',
    )

    @field_validator("confirmation")
    @classmethod
    def _confirm(cls, v: str) -> str:
        if v.strip() != "DELETE MY ACCOUNT":
            raise ValueError('Confirmation must be exactly "DELETE MY ACCOUNT".')
        return v
