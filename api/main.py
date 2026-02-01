"""
FastAPI 应用：用于本地 uvicorn 启动。
Cloudflare Workers 部署请使用 worker.py 作为入口。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.service import generate_and_review


class GenerateRequest(BaseModel):
    """POST /generate_and_review 请求体"""

    card: dict = Field(..., description="结构卡片 JSON 对象")
    n: int = Field(default=5, ge=1, le=10, description="生成变体数量")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载 .env（本地运行时）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    yield


app = FastAPI(
    title="创意评测 API",
    description="生成变体并评审，返回 PASS/REVISE/KILL 表格数据",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    """健康检查"""
    return {"status": "ok"}


@app.post("/generate_and_review")
async def post_generate_and_review(req: GenerateRequest, request: Request):
    """
    输入结构卡片 JSON + n，返回评审表格数据（含 PASS/REVISE/KILL）。
    """
    env = request.scope.get("env") if hasattr(request, "scope") else None

    result = generate_and_review(
        card_dict=req.card,
        n=req.n,
        env=env,
    )
    return result
