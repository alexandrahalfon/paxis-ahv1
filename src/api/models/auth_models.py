"""
Pydantic models for auth endpoints.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    institution: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class PatientRegisterRequest(BaseModel):
    """Patient self-registration. No institution (patients don't have one),
    and an optional invite code from their physician. Without a code the
    account is created unlinked and the patient can request a connection
    afterwards, which their physician then approves."""

    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    invite_code: Optional[str] = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    institution: Optional[str] = None
    # Defaults to physician so responses for pre-role accounts are unchanged.
    role: str = "physician"
