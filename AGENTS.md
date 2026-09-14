# SENTINEL agent conventions (SIH26143 / NTRO)

python: 3.11, mypy-strict, ruff, fastapi, async I/O, loguru (never print),
  pydantic-v2, pytest + httpx AsyncClient
typescript: strict, no any, react-18, zustand, tailwind, axios typed, vitest

rules:
- Each FastAPI service owns ONE domain; inter-service calls via sentinel-api gateway.
- Parameterized SQL only; GIST index on every geometry column; UUID PKs.
- Inference in threadpool, models loaded once via lifespan; tiled SAR inference.
- NEVER skip WMC ∇·K correction in backward SDE; always select K_ij regime first.
- Attribution shows Wilson 95% CI, never point estimates alone.
- UI dark theme only; globe is hero; every datum shows confidence.