from typing import Literal

from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: str
    name: str
    api_key: str = Field(repr=False)
    role: Literal["admin", "user"] = "user"
    created_at: str = ""


class UserCreateRequest(BaseModel):
    name: str
    role: Literal["admin", "user"] = "user"


class UserResponse(BaseModel):
    trace_id: str
    user: User


class UserListResponse(BaseModel):
    trace_id: str
    users: list[User]


class UserDeleteResponse(BaseModel):
    trace_id: str
    user_id: str
    deleted: bool
