"""
QuantMind Pro 2.0  —  完整修复版
修复清单：
  1. 数据层全部基于 stock_zh_a_hist（已测试可用），移除 spot_em 依赖
  2. 聊天布局修正：消息在上，输入框在下（正确的渲染顺序）
  3. get_stock_history_tech 完整保留
  4. 侧边栏按钮正确触发对话
  5. Python 3.9 兼容（无 dict|None 语法）
  6. DeepSeek 连接支持代理配置
  -- 新增修复 --
  7. 修复股票名显示：清洗 AKShare 名称前后的空格
  8. 修复 K 线图消失：将图表渲染槽位移出 status 块
  9. 修复 HTML 原始标签展示：强制正则清洗大模型输出的 markdown 代码块
  10. 提升分析深度：增强 Prompt 限制输出字数与逻辑拆解
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta

# ── 可选依赖（缺少时不崩溃）──
try:
    import chromadb;

    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    from pypdf import PdfReader;

    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

try:
    from tavily import TavilyClient;

    TAVILY_OK = True
except ImportError:
    TAVILY_OK = False

import openai

# ================================================================
# 1. 页面配置 & 样式
# ================================================================
st.set_page_config(page_title="QuantMind Pro 2.0", page_icon="🧿", layout="wide")


def _bj_now():
    return datetime.utcnow() + timedelta(hours=8)


CURRENT_DATE = _bj_now().strftime("%Y-%m-%d")
CURRENT_TIME = _bj_now().strftime("%H:%M")

st.markdown("""
<style>
/* ── 全局 ── */
.stApp { background:#F0F4F8; color:#1A202C; font-family:'PingFang SC','Inter',sans-serif; }

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F172A 0%,#1E293B 100%);
}
[data-testid="stSidebar"] * { color:#CBD5E1 !important; }
[data-testid="stSidebar"] h2 { color:#F8FAFC !important; }
[data-testid="stSidebar"] div.stButton > button {
    background:rgba(255,255,255,0.08) !important;
    color:#CBD5E1 !important;
    border:1px solid rgba(255,255,255,0.15) !important;
    border-radius:8px; font-weight:500; width:100%;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background:rgba(59,130,246,0.35) !important;
    color:#fff !important; border-color:#3B82F6 !important;
}

/* ── 聊天气泡 ── */
[data-testid="stChatMessageUser"] {
    background:linear-gradient(135deg,#1E40AF,#3B82F6);
    color:#fff; border-radius:16px 16px 4px 16px;
    box-shadow:0 4px 12px rgba(59,130,246,0.25);
}
[data-testid="stChatMessageAssistant"] {
    background:#fff; border:1px solid #E2E8F0;
    border-left:4px solid #3B82F6; color:#1A202C;
    border-radius:16px 16px 16px 4px;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

/* ── 研报卡片 ── */
.agent-card {
    background:#fff; border:1px solid #E2E8F0;
    border-radius:12px; padding:20px; margin:8px 0;
    box-shadow:0 2px 8px rgba(0,0,0,0.05); color:#1A202C;
}
.agent-card h4 { color:#1E40AF; font-size:15px; margin-bottom:10px; }

/* ── 指标4格 ── */
.kpi-grid {
    display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
    gap:10px; margin:12px 0;
}
.kpi-card {
    background:#fff; border:1px solid #E2E8F0;
    border-radius:10px; padding:12px 14px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.kpi-label  { font-size:11px; color:#64748B; font-weight:600;
              text-transform:uppercase; margin-bottom:5px; }
.kpi-value  { font-size:21px; font-weight:800; }
.kpi-sub    { font-size:11px; color:#94A3B8; margin-top:3px; }
.kpi-tech-label { font-size:11px; color:#1E40AF; font-weight:700;
                  text-transform:uppercase; margin-bottom:5px; }
.kpi-tech-value { font-size:17px; font-weight:800; color:#DC2626; }

/* ── 右侧摘要 ── */
.summary-float {
    background:#fff; border:1px solid #E2E8F0;
    border-top:4px solid #3B82F6; border-radius:12px;
    padding:14px; margin-bottom:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.07);
}

/* ── 极速报价 ── */
.quote-card {
    background:linear-gradient(135deg,#0F172A,#1E293B);
    border-radius:10px; padding:14px; text-align:center; margin-top:6px;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# 2. 资源初始化（带代理支持）
# ================================================================
@st.cache_resource
def _init_resources():
    col = None
    if CHROMA_OK:
        try:
            cc = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "quant_brain_v2"))
            col = cc.get_or_create_collection(name="pdf_knowledge")
        except Exception as e:
            st.warning(f"ChromaDB 初始化失败（记忆功能不可用）：{e}")

    tav = None
    if TAVILY_OK:
        try:
            tav = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        except Exception as e:
            st.warning(f"Tavily 初始化失败（联网搜索不可用）：{e}")

        # DeepSeek（支持 HTTP 代理，在 secrets.toml 中配置 HTTP_PROXY）
        ai = None
        try:
            proxy = st.secrets.get("HTTP_PROXY", None)
            if proxy:
                import httpx
                ai = openai.OpenAI(
                    api_key=st.secrets["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    # 【修改这里】：将 timeout 从 60.0 修改为 180.0（等待3分钟），增加重试次数
                    http_client=httpx.Client(proxy=proxy, timeout=180.0),
                    max_retries=3)
            else:
                ai = openai.OpenAI(
                    api_key=st.secrets["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    # 【修改这里】：将 timeout 从 60.0 修改为 180.0，增加重试次数
                    timeout=180.0,
                    max_retries=3)
        except Exception as e:
            st.error(f"DeepSeek 初始化失败：{e}")

        return col, tav, ai


_mem_col, _tavily, _ai = _init_resources()


# ================================================================
# 3. 数据层
# ================================================================

@st.cache_data(ttl=3600)
def get_stock_name_map():
    """代码→名称，缓存1小时。失败返回空字典，不影响后续查询"""
    try:
        df = ak.stock_info_a_code_name()
        c0 = df.columns[0]  # 'code'
        c1 = df.columns[1]  # 'name'
        # [修复] 增加 str.strip() 过滤隐藏空格，防止匹配失败
        df[c0] = df[c0].astype(str).str.strip().str.zfill(6)
        df[c1] = df[c1].astype(str).str.strip()
        return dict(zip(df[c0], df[c1]))
    except Exception:
        return {}


def get_stock_spot(code):
    code = str(code).strip().zfill(6)
    try:
        start = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, adjust="qfq")
        if df is None or df.empty:
            return None, f"代码 {code} 无K线数据（可能代码错误或尚未上市）"
        last = df.iloc[-1]
        name_map = get_stock_name_map()
        name = name_map.get(code, f"股票{code}")
        return {
            "代码": code,
            "名称": name,
            "最新价": float(last["收盘"]),
            "涨跌幅": float(last["涨跌幅"]),
            "成交额": float(last["成交额"]),
        }
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=1800)
def get_stock_history_tech(code):
    code = str(code).strip().zfill(6)
    try:
        start = (datetime.now() - timedelta(days=130)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, adjust="qfq")
        if df is None or df.empty:
            return None

        delta = df["收盘"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        df["MA5"] = df["收盘"].rolling(5).mean()
        df["MA20"] = df["收盘"].rolling(20).mean()

        r60 = df.tail(60)
        hi = r60["最高"].max()
        lo = r60["最低"].min()
        span = hi - lo

        return {
            "df": df.tail(60).copy(),
            "rsi": round(float(df["RSI"].iloc[-1]), 2),
            "support": round(lo + span * 0.382, 2),
            "resistance": round(lo + span * 0.618, 2),
        }
    except Exception:
        return None


# ================================================================
# 4. 工具函数
# ================================================================
def fmt_vol(val):
    try:
        v = float(val)
        if v >= 1e8: return f"{v / 1e8:.2f}亿"
        if v >= 1e4: return f"{v / 1e4:.2f}万"
        return str(round(v, 2))
    except:
        return str(val)


def ingest_pdf(f):
    if not PYPDF_OK:     return "❌ 未安装 pypdf"
    if _mem_col is None: return "❌ 向量数据库未初始化"
    try:
        text = "".join(p.extract_text() or "" for p in PdfReader(f).pages)
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
        _mem_col.add(
            documents=chunks,
            ids=[f"{f.name}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": f.name, "date": str(datetime.now())} for _ in chunks])
        return f"✅ 已学习：{f.name}（{len(chunks)} 段）"
    except Exception as e:
        return f"❌ 失败：{e}"


def search_memory(q):
    if _mem_col is None: return ""
    try:
        res = _mem_col.query(query_texts=[q], n_results=2)
        docs = res.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except:
        return ""


def web_search(q, n=3):
    if _tavily is None: return ""
    try:
        res = _tavily.search(query=q, max_results=n, search_depth="basic")
        return "\n".join(r["content"] for r in res.get("results", []))
    except:
        return ""


def ai_stream(messages, container, prefix=""):
    """流式调用 DeepSeek，并在前端过滤 Markdown 代码块"""
    if _ai is None:
        container.error("❌ AI 未初始化，请检查 DEEPSEEK_API_KEY")
        return ""

    msgs = []
    for m in messages:
        c = m.get("content", "")
        msgs.append({"role": m["role"],
                     "content": c[:2000] + "…" if len(c) > 2000 else c})

    for attempt in range(2):
        try:
            stream = _ai.chat.completions.create(
                model="deepseek-chat", messages=msgs,
                stream=True, max_tokens=2000)
            full = ""
            for chunk in stream:
                d = chunk.choices[0].delta.content
                if d:
                    full += d
                    # [修复] 实时去除非法包裹块，防止前端破损
                    display_text = re.sub(r"```html\n?|```\n?", "", full, flags=re.IGNORECASE)
                    container.markdown(prefix + display_text + "▌", unsafe_allow_html=True)

            clean_final = re.sub(r"```html\n?|```\n?", "", full, flags=re.IGNORECASE)
            container.markdown(prefix + clean_final, unsafe_allow_html=True)
            return clean_final
        except Exception as e:
            if attempt == 0:
                container.warning("⚠️ 连接超时，正在重试…")
            else:
                container.error(
                    f"❌ AI 调用失败：{e}\n\n"
                    "💡 国内访问建议在 `.streamlit/secrets.toml` 中配置代理：\n"
                    "`HTTP_PROXY = \"http://127.0.0.1:7890\"`")
    return ""


# ================================================================
# 5. Session State 初始化
# ================================================================
def _ss(key, val):
    if key not in st.session_state:
        st.session_state[key] = val


_ss("messages", [{"role": "assistant", "content":
    "您好，我是 **QuantMind Pro 2.0** 🧿\n\n"
    "**支持的指令：**\n"
    "- 直接输入股票代码（如 `600036`）→ 深度诊股 + 指标卡 + K线图\n"
    "- `对比 600519 000858` → 两只股票 PK\n"
    "- `实时内参 半导体` 或 `热点` → 行业内参 + 推荐个股\n"
    "- 任意投资问题 → AI 智能解答"}])
_ss("latest_summary", None)
_ss("pending_prompt", None)
_ss("quick_result", None)

# ================================================================
# 6. 侧边栏
# ================================================================
with st.sidebar:
    st.markdown("## 💠 QuantMind Pro 2.0")
    st.caption("Professional AI Investment Terminal")
    st.markdown(f"📅 `{CURRENT_DATE}` &nbsp;·&nbsp; 🕐 `{CURRENT_TIME}` BJ")
    st.markdown("---")

    if PYPDF_OK and CHROMA_OK:
        with st.expander("📂 研报知识库", expanded=False):
            pdf = st.file_uploader("上传研报 PDF", type="pdf", key="pdf_up")
            if pdf and st.button("导入知识库", key="btn_pdf"):
                with st.spinner("处理中…"):
                    st.success(ingest_pdf(pdf))
    else:
        st.caption("安装 chromadb & pypdf 可启用研报库")

    st.markdown("#### 🛠️ 快捷功能")
    sidebar_btns = [
        ("📉 深度诊股", "请深度诊断一只股票，我将输入代码"),
        ("🆚 股票 PK", "对比 600519 000858"),
        ("🔥 市场热点", "今日A股市场热点"),
        ("⚡ 实时内参", "实时内参 大盘"),
        ("🤖 AI策略问答", "请分析当前A股操作策略"),
    ]
    for _label, _prompt in sidebar_btns:
        if st.button(_label, key=f"sb_{_label}", use_container_width=True):
            st.session_state.pending_prompt = _prompt
            st.rerun()

    st.markdown("---")
    with st.expander("🔧 诊断", expanded=False):
        if st.button("测试数据接口", key="btn_diag"):
            with st.spinner("测试中…"):
                try:
                    _h = ak.stock_zh_a_hist(
                        symbol="600519", period="daily",
                        start_date=(datetime.now() - timedelta(days=5)).strftime("%Y%m%d"),
                        adjust="qfq")
                    st.success(f"✅ K线接口正常（{len(_h)} 行）")
                except Exception as _ex:
                    st.error(f"❌ K线接口：{_ex}")
                try:
                    _nm = ak.stock_info_a_code_name()
                    st.success(f"✅ 名称库正常（{len(_nm)} 条）")
                except Exception as _ex:
                    st.error(f"❌ 名称库：{_ex}")
                if _ai:
                    try:
                        _ai.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": "OK"}],
                            max_tokens=3, stream=False)
                        st.success("✅ DeepSeek API 连通")
                    except Exception as _ex:
                        st.error(f"❌ DeepSeek：{_ex}")
    st.caption("数据:AKShare · AI:DeepSeek · 搜索:Tavily\n仅供参考，不构成投资建议")

# ================================================================
# 7. 主布局
# ================================================================
col_chat, col_info = st.columns([7, 3])

# ── 右侧信息栏 ──
with col_info:
    st.markdown("### 📊 市场雷达")

    if st.session_state.latest_summary:
        s = st.session_state.latest_summary
        try:
            pf = float(str(s.get("pct", 0)))
        except:
            pf = 0.0
        pc = "#DC2626" if pf >= 0 else "#059669"
        st.markdown(f"""
        <div class="summary-float">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
            <div>
              <div style="font-size:16px;font-weight:800;color:#0F172A">{s["name"]}</div>
              <span style="background:#F1F5F9;padding:1px 8px;border-radius:10px;
                    font-size:11px;color:#64748B">{s["code"]}</span>
            </div>
            <div style="text-align:right;">
              <div style="font-size:22px;font-weight:800;color:{pc}">{s["price"]}</div>
              <div style="font-size:12px;font-weight:600;color:{pc}">{pf:+.2f}%</div>
            </div>
          </div>
          <hr style="border-top:1px solid #F1F5F9;margin:8px 0;">
          <div style="display:flex;gap:12px;font-size:12px;flex-wrap:wrap;">
            <span style="color:#64748B">RSI: <b style="color:#1E293B">{s["rsi"]}</b></span>
            <span style="color:#64748B">压力: <b style="color:#DC2626">{s["res"]}</b></span>
            <span style="color:#64748B">支撑: <b style="color:#059669">{s["sup"]}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👋 输入股票代码后，诊断摘要将在此显示。")

    st.markdown("#### ⚡ 极速报价")
    qcode = st.text_input("", placeholder="代码如 600036",
                          key="qi", label_visibility="collapsed")
    if st.button("查 询", key="btn_q", use_container_width=True):
        if qcode and qcode.strip():
            with st.spinner("查询中…"):
                _qr = get_stock_spot(qcode.strip())
                q_data = _qr[0] if isinstance(_qr, tuple) else _qr
                q_err = _qr[1] if (isinstance(_qr, tuple) and _qr[0] is None) else None
            st.session_state.quick_result = ("ERR", q_err) if q_err else q_data

    if st.session_state.quick_result:
        qr = st.session_state.quick_result
        if isinstance(qr, tuple):
            st.warning(f"⚠️ {qr[1]}")
        else:
            try:
                qp = float(str(qr.get("涨跌幅", 0)))
            except:
                qp = 0.0
            qc = "#EF4444" if qp >= 0 else "#10B981"
            st.markdown(f"""
            <div class="quote-card">
              <div style="font-size:12px;color:#94A3B8;margin-bottom:4px;">
                {qr.get("名称", "—")} · {qr.get("代码", "")}
              </div>
              <div style="font-size:26px;font-weight:800;color:{qc}">{qr.get("最新价", "N/A")}</div>
              <div style="font-size:13px;font-weight:600;color:{qc};margin-top:2px;">{qp:+.2f}%</div>
              <div style="font-size:11px;color:#64748B;margin-top:5px;">
                成交额 {fmt_vol(qr.get("成交额", 0))}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── 左侧聊天区 ──
with col_chat:
    st.markdown("### 💬 智能分析师")

    # ── step1：渲染历史消息 ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            c = msg["content"]
            if any(t in c for t in ["<div", "<table", "<span"]):
                # [修复] 双重保险：历史记录中如果存在残留的 markdown 代码块，强行去除
                c = re.sub(r"```html\n?|```\n?", "", c, flags=re.IGNORECASE)
                st.markdown(c, unsafe_allow_html=True)
            else:
                st.markdown(c)

    # ── step2：处理侧边栏快捷按钮 pending_prompt ──
    _auto_prompt = None
    if st.session_state.pending_prompt:
        _auto_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # ── step3：接收输入 ──
    _typed = st.chat_input(
        "输入股票代码 / '对比 600519 000858' / '实时内参 半导体' / 直接提问…")

    prompt = (_typed or _auto_prompt or "").strip()

    # ── step4：处理新输入 ──
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            out_box = st.empty()
            chart_box = st.empty()  # [修复] 单独为K线图开辟外部占位符，跳出 status 块域
            full_resp = ""
            _need_rerun = False

            code_match = re.search(r"\b(\d{6})\b", prompt)
            is_compare = any(kw in prompt for kw in ["对比", "pk", "PK", "比较", "VS", "vs"])
            is_neican = any(kw in prompt for kw in ["内参", "热点", "热门", "涨停", "板块"])

            # =========================================================
            # 逻辑 A：单股深度诊断
            # =========================================================
            if code_match and not is_compare and not is_neican:
                code = code_match.group(1)

                with st.status(f"🔍 深度分析 {code}…", expanded=True) as sts:
                    sts.write("🚀 获取实时行情…")
                    _sr = get_stock_spot(code)
                    spot = _sr[0] if isinstance(_sr, tuple) else _sr
                    _serr = _sr[1] if isinstance(_sr, tuple) else "数据源异常"

                    if spot is None:
                        out_box.error(
                            f"❌ 获取 **{code}** 失败\n\n"
                            f"原因：{_serr}\n\n"
                            "建议：\n"
                            "1. 确认代码正确（沪市6开头，深市0开头）\n"
                            "2. `pip install akshare --upgrade`\n"
                            "3. 侧边栏「🔧 诊断」→「测试数据接口」查看详情")
                        full_resp = f"代码 {code} 数据获取失败：{_serr}"
                    else:
                        sts.write("📈 计算技术指标…")
                        tech = get_stock_history_tech(code)

                        sts.write("🌐 检索市场资讯…")
                        news = web_search(f"{CURRENT_DATE} {spot['名称']} 股票 利好利空", n=2)
                        if not news: news = "暂无联网资讯"

                        sts.write("🧠 调取研报记忆…")
                        mem = search_memory(f"{spot['名称']} 投资观点")
                        if not mem: mem = "暂无本地研报"

                        sts.update(label="✨ AI 生成研报中…", state="complete")

                        price_v = spot.get("最新价", "N/A")
                        vol_v = fmt_vol(spot.get("成交额", 0))
                        rsi_v = tech["rsi"] if tech else "N/A"
                        sup_v = tech["support"] if tech else "N/A"
                        res_v = tech["resistance"] if tech else "N/A"
                        try:
                            pct_f = float(str(spot.get("涨跌幅", 0)))
                        except:
                            pct_f = 0.0
                        pc = "#DC2626" if pct_f >= 0 else "#059669"
                        rsi_zone = "中性区间"
                        try:
                            rv = float(str(rsi_v))
                            if rv > 70:
                                rsi_zone = "⚠️ 超买"
                            elif rv < 30:
                                rsi_zone = "✅ 超卖"
                        except:
                            pass

                        metrics_html = f"""
<div style="margin:2px 0 8px 0;">
  <div style="font-size:17px;font-weight:800;color:#0F172A;margin-bottom:10px;">
    {spot["名称"]}
    <span style="background:#F1F5F9;padding:2px 9px;border-radius:10px;
          font-size:11px;color:#64748B;margin-left:6px;font-weight:400;">{code}</span>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">现价</div>
      <div class="kpi-value" style="color:{pc}">{price_v}</div>
      <div class="kpi-sub" style="color:{pc};font-weight:600">{pct_f:+.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">成交额</div>
      <div class="kpi-value" style="font-size:18px;color:#1E293B">{vol_v}</div>
      <div class="kpi-sub">今日</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-tech-label">支撑 / 压力</div>
      <div class="kpi-tech-value">{sup_v} / {res_v}</div>
      <div class="kpi-sub">斐波那契 0.382/0.618</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-tech-label">RSI(14)</div>
      <div class="kpi-tech-value">{rsi_v}</div>
      <div class="kpi-sub">{rsi_zone}</div>
    </div>
  </div>
</div>"""
                        out_box.markdown(metrics_html, unsafe_allow_html=True)

                        # [修复] 渲染到单独的图表容器，防止被 status 抹除
                        if tech and tech.get("df") is not None:
                            df_c = tech["df"].copy()
                            df_c["日期"] = pd.to_datetime(df_c["日期"])
                            chart_box.line_chart(
                                df_c.set_index("日期")[["收盘", "MA5", "MA20"]],
                                color=["#EF4444", "#7DD3FC", "#1D4ED8"])

                        ai_box = st.empty()
                        # [修复] 加强 Prompt 输出逻辑拆解，且强制输出纯 HTML
                        ai_resp = ai_stream([{"role": "user", "content":
                            f"你是资深A股分析师，风格专业且深度极强。\n"
                            f"股票：{spot['名称']}（{code}），现价={price_v}，"
                            f"涨跌={pct_f:+.2f}%，RSI={rsi_v}，"
                            f"支撑={sup_v}，压力={res_v}，成交额={vol_v}\n"
                            f"资讯：{news[:600]}\n本地研报：{mem[:300]}\n\n"
                            "输出HTML研报，外层<div class='agent-card'>，包含：\n"
                            "①【核心判断】看多/看空/震荡结论及核心逻辑（不少于50字）\n"
                            "②【技术面深度剖析】结合RSI、均线、支撑压力位，深度分析当前量价配合及多空博弈状态（不少于100字）\n"
                            "③【消息面与基本面】结合资讯深度解读对股价的实质性影响与情绪催化（不少于100字）\n"
                            "④【操盘建议与推演】具体建仓/减仓/止损点位及未来一周走势推演（不少于100字）\n"
                            "要求深色文字(#1A202C)，禁止白色文字。\n"
                            "【重要警告】直接输出纯HTML代码，绝不允许使用 ```html 等markdown代码块包裹！"}],
                                            ai_box)

                        full_resp = metrics_html + "\n\n" + ai_resp

                        st.session_state.latest_summary = {
                            "name": spot["名称"], "code": code,
                            "price": price_v, "pct": pct_f,
                            "rsi": rsi_v, "res": res_v,
                            "sup": sup_v}
                        _need_rerun = True

            # =========================================================
            # 逻辑 B：股票 PK
            # =========================================================
            elif is_compare:
                codes = re.findall(r"\b(\d{6})\b", prompt)
                if len(codes) < 2:
                    out_box.info("请提供两个六位数代码，例如：对比 600519 000858")
                    full_resp = "请提供两个代码进行对比。"
                else:
                    ca, cb = codes[0], codes[1]
                    with st.spinner("获取双股行情…"):
                        _ra = get_stock_spot(ca)
                        _rb = get_stock_spot(cb)
                        sa = _ra[0] if isinstance(_ra, tuple) else _ra
                        sb = _rb[0] if isinstance(_rb, tuple) else _rb
                        ta = get_stock_history_tech(ca)
                        tb = get_stock_history_tech(cb)

                    if sa is None or sb is None:
                        miss = ca if sa is None else cb
                        _merr = (_ra[1] if isinstance(_ra, tuple) else "") if sa is None else (
                            _rb[1] if isinstance(_rb, tuple) else "")
                        out_box.error(f"❌ 获取代码 {miss} 失败：{_merr}")
                        full_resp = f"代码 {miss} 数据获取失败。"
                    else:
                        def _p(s):
                            try:
                                return float(str(s.get("涨跌幅", 0)))
                            except:
                                return 0.0


                        pa, pb = _p(sa), _p(sb)
                        ca_c = "#DC2626" if pa >= 0 else "#059669"
                        cb_c = "#DC2626" if pb >= 0 else "#059669"

                        tbl = f"""
<div class="agent-card">
  <h4>⚔️ {sa["名称"]} vs {sb["名称"]}</h4>
  <table style="width:100%;border-collapse:collapse;font-size:13px;color:#1A202C;">
    <thead><tr style="background:#F8FAFC;">
      <th style="padding:9px;text-align:left;color:#64748B;border-bottom:2px solid #E2E8F0;">指标</th>
      <th style="padding:9px;text-align:center;color:#1E40AF;border-bottom:2px solid #E2E8F0;">
        {sa["名称"]}<br><small style="color:#94A3B8;font-weight:400">{ca}</small></th>
      <th style="padding:9px;text-align:center;color:#1E40AF;border-bottom:2px solid #E2E8F0;">
        {sb["名称"]}<br><small style="color:#94A3B8;font-weight:400">{cb}</small></th>
    </tr></thead>
    <tbody>
      <tr>
        <td style="padding:8px;color:#64748B;border-bottom:1px solid #F1F5F9">现价</td>
        <td style="padding:8px;text-align:center;font-weight:700;border-bottom:1px solid #F1F5F9">{sa.get("最新价", "N/A")}</td>
        <td style="padding:8px;text-align:center;font-weight:700;border-bottom:1px solid #F1F5F9">{sb.get("最新价", "N/A")}</td>
      </tr>
      <tr style="background:#FAFAFA">
        <td style="padding:8px;color:#64748B;border-bottom:1px solid #F1F5F9">涨跌幅</td>
        <td style="padding:8px;text-align:center;font-weight:700;color:{ca_c};border-bottom:1px solid #F1F5F9">{pa:+.2f}%</td>
        <td style="padding:8px;text-align:center;font-weight:700;color:{cb_c};border-bottom:1px solid #F1F5F9">{pb:+.2f}%</td>
      </tr>
      <tr>
        <td style="padding:8px;color:#64748B;border-bottom:1px solid #F1F5F9">成交额</td>
        <td style="padding:8px;text-align:center;border-bottom:1px solid #F1F5F9">{fmt_vol(sa.get("成交额", 0))}</td>
        <td style="padding:8px;text-align:center;border-bottom:1px solid #F1F5F9">{fmt_vol(sb.get("成交额", 0))}</td>
      </tr>
      <tr style="background:#FAFAFA">
        <td style="padding:8px;color:#64748B;border-bottom:1px solid #F1F5F9">RSI(14)</td>
        <td style="padding:8px;text-align:center;font-weight:600;border-bottom:1px solid #F1F5F9">{ta["rsi"] if ta else "N/A"}</td>
        <td style="padding:8px;text-align:center;font-weight:600;border-bottom:1px solid #F1F5F9">{tb["rsi"] if tb else "N/A"}</td>
      </tr>
      <tr>
        <td style="padding:8px;color:#64748B">压力位</td>
        <td style="padding:8px;text-align:center;color:#DC2626;font-weight:600">{ta["resistance"] if ta else "N/A"}</td>
        <td style="padding:8px;text-align:center;color:#DC2626;font-weight:600">{tb["resistance"] if tb else "N/A"}</td>
      </tr>
    </tbody>
  </table>
</div>"""
                        out_box.markdown(tbl, unsafe_allow_html=True)
                        ai_b = st.empty()
                        # [修复] 提升对比内容的广度
                        ai_txt = ai_stream([{"role": "user", "content":
                            f"对比 {sa['名称']}(RSI:{ta['rsi'] if ta else 'N/A'},涨跌{pa:+.2f}%) "
                            f"与 {sb['名称']}(RSI:{tb['rsi'] if tb else 'N/A'},涨跌{pb:+.2f}%)。\n"
                            "从短期弹性、资金面、技术位三个维度进行深度对比分析，给出明确的操作建议和优先推荐标的（不少于300字）。\n"
                            "要求深色文字(#1A202C)。直接输出，【重要警告】绝不允许使用 ```html 等markdown代码块包裹！"}],
                                           ai_b)
                        full_resp = tbl + "\n\n" + ai_txt

            # =========================================================
            # 逻辑 C：实时内参
            # =========================================================
            elif is_neican:
                m_ = re.search(r"内参\s*(.+)", prompt)
                topic = m_.group(1).strip() if m_ else ("大盘热门" if "热点" in prompt else "大盘")

                with st.status(f"📡 扫描 [{topic}] 最新情报…", expanded=True) as sts:
                    sts.write("🌐 检索宏观/政策…")
                    ctx1 = web_search(f"{CURRENT_DATE} {topic} A股 行业利好 政策", n=3)
                    sts.write("📊 检索板块异动与龙头…")
                    ctx2 = web_search(f"{CURRENT_DATE} A股 {topic} 涨停 龙头股", n=3)
                    if not ctx1 and not ctx2:
                        ctx1 = "（联网搜索不可用，基于训练数据分析）"
                    sts.update(label="✨ AI 内参生成中…", state="complete")

                # [修复] 加深深度，重塑要求，硬阻挡 HTML markdown
                full_resp = ai_stream([{"role": "user", "content":
                    f"你是顶级A股内参分析师，分析风格极具深度和洞察力。今日{CURRENT_DATE}，主题：{topic}\n\n"
                    f"宏观/政策：\n{ctx1[:800]}\n\n板块/涨停：\n{ctx2[:800]}\n\n"
                    "生成专业详实的内参报告，HTML格式，外层<div class='agent-card'>：\n"
                    "<h4>📌 核心资讯深度解读</h4>深度剖析3条核心利空/利好逻辑及其对资金的吸引力（不少于150字）\n"
                    "<h4>🔥 热门板块爆发逻辑</h4>详述2-3个领涨板块的驱动事件及持续性预期（不少于150字）\n"
                    "<h4>💎 重点龙头个股</h4>推荐3-5只核心标的（格式：名称[代码]），详细说明基本面与资金面异动理由\n"
                    "<h4>⚠️ 风险与规避提示</h4>指出潜在的宏观或板块回调风险点（不少于100字）\n"
                    "深色文字(#1A202C)，禁止白色文字。\n"
                    "【重要警告】直接输出纯HTML代码，绝不要使用 ```html 等markdown代码块包裹！"}],
                                      out_box)

            # =========================================================
            # 逻辑 D：通用问答（带历史上下文）
            # =========================================================
            else:
                hist_msgs = [
                    {"role": m["role"],
                     "content": re.sub(r"<[^>]+>", "", m["content"])[:600]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ("user", "assistant")
                ]
                hist_msgs.insert(0, {"role": "system", "content":
                    f"你是专注A股的智能投资助手，当前日期{CURRENT_DATE}。"
                    "回答专业简练，适当使用Markdown，避免白色文字。"})
                full_resp = ai_stream(hist_msgs, out_box)

            if full_resp:
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_resp})

            if _need_rerun:
                st.rerun()