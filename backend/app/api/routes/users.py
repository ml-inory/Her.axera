from typing import Annotated

from fastapi import APIRouter, Body, Depends

from app.api.deps import bind_trace_id
from app.models.user import UserCreateRequest, UserDeleteResponse, UserListResponse, UserResponse
from app.services.user_service import user_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=UserListResponse)
async def list_users(trace_id: Annotated[str, Depends(bind_trace_id)]) -> UserListResponse:
    return UserListResponse(trace_id=trace_id, users=user_service.list_users())


@router.post("", response_model=UserResponse)
async def create_user(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    request: Annotated[UserCreateRequest, Body()],
) -> UserResponse:
    user = user_service.create_user(name=request.name, role=request.role)
    return UserResponse(trace_id=trace_id, user=user)


@router.delete("/{user_id}", response_model=UserDeleteResponse)
async def delete_user(
    user_id: str,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> UserDeleteResponse:
    deleted = user_service.delete_user(user_id)
    return UserDeleteResponse(trace_id=trace_id, user_id=user_id, deleted=deleted)
