# 部署指南

## 一、推送到 GitHub

**方式一：双击运行** `push_to_github.bat`

**方式二：命令行**

```bash
cd creative_eval_demo
git init
git remote add origin https://github.com/MyraWang0406/AIGC-auto-ads.git
git add .
git commit -m "Initial commit: AIGC 创意评测决策看板"
git branch -M main
git push -u origin main
```

> 推送时需输入 GitHub 用户名和 Personal Access Token（密码已不支持）。

---

## 二、关于 Cloudflare

**Streamlit 无法部署到 Cloudflare Pages / Workers**（仅支持静态站点）：
- 推荐使用 **Streamlit Community Cloud** 部署（见下方）
- 或用 Cloudflare Tunnel 暴露自建服务器上的 Streamlit

---

## 三、Streamlit Community Cloud 部署（推荐）

1. 打开 https://share.streamlit.io
2. 用 GitHub 登录
3. **New app** → 选择仓库 `MyraWang0406/AIGC-auto-ads`
4. **Main file path**：`app_demo.py`
5. **Deploy**

得到链接：`https://<app-name>.streamlit.app`
