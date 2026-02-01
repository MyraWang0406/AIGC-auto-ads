# AIGC 创意评测决策看板 (AIGC-auto-ads)

基于 Streamlit 的投放素材评测与决策看板 Demo，支持结构卡片、OFAAT 变体生成、门禁评测与元素级贡献分析。

## 本地运行

```bash
cd creative_eval_demo
pip install -r requirements.txt
streamlit run app_demo.py
```

默认端口 3100（见 `.streamlit/config.toml`），或指定：

```bash
streamlit run app_demo.py --server.port 8501
```

## 云部署（Streamlit Community Cloud）

> **说明**：Streamlit 是 Python 服务端应用，**不能部署到 Cloudflare Pages**（仅支持静态站点）。推荐使用 **Streamlit Community Cloud** 免费部署。

### 步骤

1. 将本仓库推送到 GitHub
2. 打开 [share.streamlit.io](https://share.streamlit.io)
3. 用 GitHub 登录 → **New app**
4. 选择仓库 `MyraWang0406/AIGC-auto-ads`
5. 分支 `main`
6. **Main file path**: `app_demo.py`
7. **App URL**：可选自定义子域名
8. 点击 **Deploy**

部署完成后会得到类似 `https://xxx.streamlit.app` 的链接。

## 项目结构

```
creative_eval_demo/
├── app_demo.py          # 主入口（决策看板 Demo）
├── requirements.txt
├── samples/             # 示例 JSON
├── ui/                  # 样式
└── ...
```

## 联系

myrawzm0406@163.com
