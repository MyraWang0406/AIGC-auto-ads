# Streamlit 云部署排错

## "Error running app" 时请检查

### 1. Main file path 是否正确

根据仓库结构二选一：
- **若根目录有 `app_demo.py`**：Main file path = `app_demo.py`
- **若根目录有 `creative_eval_demo/` 文件夹**：Main file path = `creative_eval_demo/app_demo.py`

### 2. 重新部署后查看具体错误

已在 app_demo.py 中加入错误捕获，若导入或运行出错，页面会显示详细 traceback。请截图发给我。

### 3. 常见问题

| 现象 | 处理 |
|------|------|
| 导入失败 | 检查 requirements.txt 是否完整，Streamlit Cloud 会自动 pip install |
| 文件找不到 | 确认 samples/ 和 vertical_config.json 已提交到 Git |
| 资源超限 | Streamlit 免费版有内存限制，可尝试减少评测集卡片数量 |

### 4. 本地先验证

```bash
cd creative_eval_demo
pip install -r requirements.txt
streamlit run app_demo.py
```

本地能跑通再部署。
