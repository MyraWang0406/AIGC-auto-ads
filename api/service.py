"""
业务逻辑封装：生成 + 评审，可注入 API Key / Model 配置。
供 FastAPI 与 Worker 复用，支持从 env 或参数传入密钥。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保项目根目录在 path 中（Worker 部署时可能需此）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import os
from typing import Any

from openrouter_client import JsonParseError, chat_completion_json
from prompts import build_generation_prompt, build_review_prompt
from schemas import (
    CreativeCard,
    CreativeVariant,
    ReviewResponse,
    ReviewResult,
    VariantWithReview,
)
from scoring import compute_fuse_decision


def _get_config(env: Any | None = None) -> tuple[str, str]:
    """获取 API Key 和 Model，优先从 env（Worker）取"""
    api_key = ""
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    if env is not None and hasattr(env, "OPENROUTER_API_KEY"):
        api_key = getattr(env, "OPENROUTER_API_KEY", "") or ""
    if env is not None and hasattr(env, "OPENROUTER_MODEL"):
        model = getattr(env, "OPENROUTER_MODEL", model) or model
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    return api_key, model


def _patch_env_for_request(api_key: str, model: str) -> None:
    """临时设置环境变量供 openrouter_client 使用"""
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    if model:
        os.environ["OPENROUTER_MODEL"] = model


def run_generation(
    card: CreativeCard,
    n: int,
    env: Any | None = None,
) -> list[CreativeVariant]:
    """生成 N 个变体"""
    api_key, model = _get_config(env)
    _patch_env_for_request(api_key, model)

    prompt = build_generation_prompt(card, n=n)
    try:
        out, _ = chat_completion_json(
            [{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=8192,
            return_raw=True,
        )
        if isinstance(out, list):
            variants_data = out
        else:
            variants_data = out.get("variants", [])

        return [CreativeVariant.model_validate(v) for v in variants_data]
    except JsonParseError:
        return []
    except Exception:
        return []


def run_review(
    card: CreativeCard,
    variants: list[CreativeVariant],
    env: Any | None = None,
) -> list[ReviewResult]:
    """评审变体"""
    api_key, model = _get_config(env)
    _patch_env_for_request(api_key, model)

    if not variants:
        return []

    prompt = build_review_prompt(card, variants)
    try:
        out, _ = chat_completion_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=8192,
            return_raw=True,
        )
        resp = ReviewResponse.model_validate(out)
        results = resp.results

        out_list: list[ReviewResult] = []
        for i, v in enumerate(variants):
            rid = getattr(v, "variant_id", "") or f"v{i+1:03d}"
            r = next((x for x in results if (x.variant_id or "").strip() == rid), None)
            if r is None and i < len(results):
                r = results[i]
            if r is None:
                r = ReviewResult(variant_id=rid)
            out_list.append(r)

        while len(out_list) < len(variants):
            out_list.append(
                ReviewResult(
                    variant_id=getattr(variants[len(out_list)], "variant_id", "")
                    or f"v{len(out_list)+1:03d}"
                )
            )
        return out_list[: len(variants)]

    except JsonParseError:
        return [
            ReviewResult(error="LLM 评审结果 JSON 解析失败")
            for _ in variants
        ]
    except Exception:
        return [ReviewResult() for _ in variants]


def generate_and_review(
    card_dict: dict[str, Any],
    n: int = 5,
    env: Any | None = None,
) -> dict[str, Any]:
    """
    主入口：解析卡片 → 生成 → 评审 → 熔断决策。
    返回可供前端表格与导出使用的结构化数据。
    """
    try:
        card = CreativeCard.model_validate(card_dict)
    except Exception as e:
        return {
            "ok": False,
            "error": f"结构卡片解析失败: {e}",
            "table": [],
            "csv": "",
            "markdown": "",
        }

    api_key, _ = _get_config(env)
    if not api_key:
        return {
            "ok": False,
            "error": "请设置 OPENROUTER_API_KEY",
            "table": [],
            "csv": "",
            "markdown": "",
        }

    variants = run_generation(card, n=n, env=env)
    if not variants:
        return {
            "ok": False,
            "error": "生成变体失败或无结果",
            "table": [],
            "csv": "",
            "markdown": "",
        }

    reviews = run_review(card, variants, env=env)
    rows: list[VariantWithReview] = []
    for v, r in zip(variants, reviews):
        verdict, wt_risk, fuse = compute_fuse_decision(card, v, r)
        rows.append(
            VariantWithReview(
                variant=v,
                review=r,
                verdict=verdict,
                white_traffic_risk_final=wt_risk,
                fuse_level=fuse,
            )
        )

    # 构建表格数据（供前端渲染）
    table = []
    for i, rw in enumerate(rows, 1):
        s = rw.review.scores
        table.append({
            "index": i,
            "variant_id": rw.variant.variant_id or rw.variant.headline or rw.variant.title or "-",
            "headline": (rw.variant.headline or rw.variant.title or rw.variant.hook_type or "-")[:50],
            "decision": rw.verdict,
            "fuse_level": rw.fuse_level,
            "white_traffic_risk_final": rw.white_traffic_risk_final,
            "clarity": s.clarity,
            "hook_strength": s.hook_strength,
            "compliance_safety": s.compliance_safety,
            "expected_test_value": s.expected_test_value,
            "summary": (rw.review.error or rw.review.overall_summary or "-")[:80],
        })

    # 导出 CSV / Markdown
    from exporters import export_csv, export_markdown

    csv_str = export_csv(rows)
    md_str = export_markdown(rows)

    return {
        "ok": True,
        "table": table,
        "csv": csv_str,
        "markdown": md_str,
    }
