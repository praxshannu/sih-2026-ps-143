from __future__ import annotations

from datetime import UTC

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from app.middleware.auth import get_current_user
from app.schemas import (
    AnnotationCreate,
    AnnotationResponse,
    CaseCreate,
    CaseListResponse,
    CaseResponse,
    CaseStatus,
    CaseUpdate,
    TokenPayload,
)

router = APIRouter(prefix="/cases", tags=["cases"])

# In-memory store — replace with database in production
_cases: dict[str, dict] = {}
_annotations: dict[str, list[dict]] = {}
_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"case-{_counter:06d}"


@router.get("", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: CaseStatus | None = Query(None, alias="status"),
    _user: TokenPayload = Depends(get_current_user),
):
    cases = list(_cases.values())
    if status_filter:
        cases = [c for c in cases if c["status"] == status_filter]
    cases.sort(key=lambda c: c["updated_at"], reverse=True)
    total = len(cases)
    start = (page - 1) * page_size
    end = start + page_size
    return CaseListResponse(
        cases=[CaseResponse(**c) for c in cases[start:end]],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str, _user: TokenPayload = Depends(get_current_user)):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return CaseResponse(**case)


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(body: CaseCreate, _user: TokenPayload = Depends(get_current_user)):
    from datetime import datetime

    now = datetime.now(UTC)
    case_id = _next_id()
    case = {
        "id": case_id,
        "name": body.name,
        "description": body.description,
        "status": CaseStatus.NEW,
        "priority": body.priority,
        "region": body.region,
        "annotations": [],
        "created_at": now,
        "updated_at": now,
    }
    _cases[case_id] = case
    _annotations[case_id] = []
    logger.info("Case created: {} ({})", case_id, body.name)
    return CaseResponse(**case)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    body: CaseUpdate,
    _user: TokenPayload = Depends(get_current_user),
):
    from datetime import datetime

    case = _cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    update_data = body.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    case.update(update_data)
    case["updated_at"] = datetime.now(UTC)
    return CaseResponse(**case)


@router.post(
    "/{case_id}/annotations", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED
)
async def add_annotation(
    case_id: str,
    body: AnnotationCreate,
    _user: TokenPayload = Depends(get_current_user),
):
    from datetime import datetime

    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found")
    now = datetime.now(UTC)
    annotation = {
        "id": f"ann-{len(_annotations.get(case_id, [])) + 1:06d}",
        "case_id": case_id,
        "author": body.author,
        "text": body.text,
        "tags": body.tags,
        "created_at": now,
    }
    _annotations.setdefault(case_id, []).append(annotation)
    _cases[case_id]["updated_at"] = now
    logger.info("Annotation added to case {}: {}", case_id, annotation["id"])
    return AnnotationResponse(**annotation)
