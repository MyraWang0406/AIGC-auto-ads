"""
决策看板 UI 样式：集中管理 CSS 注入。
产品化观感：隐藏 Streamlit 干扰、蓝系、紧凑布局。
"""


def get_global_styles() -> str:
    """返回全局样式 CSS，用于 st.markdown(..., unsafe_allow_html=True)"""
    return """
<style>
/* ===== 隐藏 Streamlit 顶部工具栏 / Deploy / Menu（多选择器兜底） ===== */
header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
.stDeployButton { display: none !important; }
button[kind="header"] { display: none !important; }
footer { visibility: hidden !important; }
[data-testid="stAppViewContainer"] > header { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
/* 兼容不同版本 */
[data-testid="stAppToolbar"], [data-testid="stToolbar"], .stAppToolbar { display: none !important; }
[aria-label="Main menu"] { display: none !important; }

/* ===== 减少顶部空白，内容贴近 Header ===== */
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

/* ===== 标题区：深浅蓝渐变 + 水波纹理 ===== */
.title-banner {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 35%, #1976d2 65%, #1565c0 100%);
    background-size: 200% 200%;
    animation: wave 6s ease-in-out infinite;
    padding: 0.9rem 1.5rem;
    margin: -1rem -1rem 0.5rem -1rem;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 2px 12px rgba(21,101,192,0.2);
}
@keyframes wave {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.title-banner .title-text { color: #fff !important; font-weight: 600; font-size: 1.3rem !important; }
.title-banner { position: sticky !important; top: 0 !important; z-index: 100 !important; }

/* ===== 联系作者：右下角固定，黑底白字 ===== */
.contact-footer {
    position: fixed !important; bottom: 0 !important; right: 0 !important;
    background: #1a1a1a !important; color: #fff !important;
    padding: 0.4rem 0.8rem !important; font-size: 0.85rem !important;
    border-radius: 8px 0 0 0 !important; z-index: 9999 !important;
}
.contact-footer a { color: #fff !important; text-decoration: none !important; }

/* ===== 全站蓝色系，禁止装饰红 ===== */
button[kind="primary"] { background-color: #2563eb !important; }
span[data-baseweb="tag"] { background-color: #3b82f6 !important; color: #fff !important; border-color: #2563eb !important; }
span[data-baseweb="tag"] span { color: #fff !important; }

/* ===== 侧边栏电梯导航：固定左侧，字小 ===== */
[data-testid="stSidebar"] { min-width: 90px !important; max-width: 140px !important; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { font-size: 0.74rem !important; }
[data-testid="stSidebar"] button { font-size: 0.7rem !important; padding: 0.3rem 0.4rem !important; }
.elevator-title { font-size: 0.75rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }

/* ===== 元素贡献卡片 ===== */
.elem-card { border: 1px solid #90caf9; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: #f8fbff; }
.elem-card.pull { border-left: 4px solid #1976d2; }
.elem-card.drag { border-left: 4px solid #64b5f6; }
.elem-card.unknown { border-left: 4px solid #90a4ae; }

/* ===== 实验工单卡片 ===== */
.suggest-card { border: 1px solid #90caf9; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: #fafcff; }
.suggest-card .field { font-size: 0.9rem; margin: 0.25rem 0; }
.suggest-card .value { font-weight: 600; color: #1565c0; }

/* ===== 多选下拉：选项完整显示，字号调小 ===== */
[data-baseweb="popover"] { min-width: 320px !important; max-width: 400px !important; }
ul[role="listbox"] li { white-space: normal !important; word-break: break-word !important; padding: 0.5rem !important; font-size: 0.75rem !important; }
[data-baseweb="menu"] li { white-space: normal !important; word-break: break-word !important; font-size: 0.75rem !important; }
[data-baseweb="tag"] { max-width: none !important; white-space: normal !important; font-size: 0.72rem !important; }
span[data-baseweb="tag"] span { font-size: 0.72rem !important; }
div[data-baseweb="select"] > div { min-width: 180px !important; }

/* ===== 结构卡摘要：数值字号 ===== */
[data-testid="stMetric"] > div { font-size: 0.78rem !important; }
[data-testid="stMetric"] label { font-size: 0.75rem !important; }

/* ===== 表格 ===== */
[data-testid="stDataFrame"], .stDataFrame { overflow-x: auto !important; max-width: 100%; }
</style>
"""
