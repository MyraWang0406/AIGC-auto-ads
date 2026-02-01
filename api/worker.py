"""
Cloudflare Workers 入口：将 FastAPI 应用挂载到 WorkerEntrypoint。
部署时 wrangler 的 main 指向此文件。
"""
from __future__ import annotations

import asgi
from workers import WorkerEntrypoint

from api.main import app


class Default(WorkerEntrypoint):
    """Cloudflare Worker 入口"""

    async def fetch(self, request):
        return await asgi.fetch(app, request, self.env)
