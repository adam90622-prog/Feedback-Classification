"""
고객 피드백 모니터링 시스템 — Streamlit 버전
CSV 업로드 → LLM/룰기반 분류 → 대시보드 + 완료 처리
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px
from classifier import classify, get_type_counts, RISK_CONFIG, SECONDARY_CONFIG

# ── 페이지 설정 ───────────────────────────────────────────────────

st.set_page_config(
    page_title="☕ 피드백 모니터링",
    page_icon="☕",
    layout="wide",
)

st.markdown("""
<style>
/* 완료된 카드 스타일 */
.done-card {
    opacity: 0.45;
    background: #f5f5f5 !important;
    border-left-color: #bbb !important;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 5px solid #bbb;
}
.done-card * { color: #999 !important; }
/* 활성 Top3 카드 */
.active-card-1 { background:#FF4B4B12; border-left:6px solid #FF4B4B;
                 border-radius:12px; padding:16px 20px; margin-bottom:10px; }
.active-card-2 { background:#FF8C0012; border-left:6px solid #FF8C00;
                 border-radius:12px; padding:16px 20px; margin-bottom:10px; }
.active-card-3 { background:#FFC30012; border-left:6px solid #FFC300;
                 border-radius:12px; padding:16px 20px; margin-bottom:10px; }
.card-rank   { font-size:1.05rem; font-weight:800; }
.card-quote  { font-size:1.02rem; font-weight:600; margin:6px 0 4px; }
.card-reason { font-size:0.85rem; color:#666; }
.card-meta   { font-size:0.78rem; color:#aaa; margin-top:3px; }
/* 완료 배지 */
.done-badge { display:inline-block; background:#e0fbe0; color:#1a8a3a;
              border-radius:10px; padding:2px 10px; font-size:0.8rem;
              font-weight:700; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

# ── 세션 상태 초기화 ──────────────────────────────────────────────

if "completed" not in st.session_state:
    st.session_state.completed = set()   # 완료된 피드백 id 집합
if "enriched" not in st.session_state:
    st.session_state.enriched  = None
if "mode" not in st.session_state:
    st.session_state.mode      = ""
if "last_file" not in st.session_state:
    st.session_state.last_file = ""

# ── 컬럼 자동 감지 ────────────────────────────────────────────────

CONTENT_ALIASES = ["내용","content","피드백","feedback","comment","텍스트","text","메시지"]
RATING_ALIASES  = ["별점","rating","score","점수","평점","stars"]
DATE_ALIASES    = ["날짜","date","received_at","받은날짜","일자"]
CHANNEL_ALIASES = ["경로","channel","source","채널","출처"]

def _detect(cols, aliases):
    for a in aliases:
        for c in cols:
            if a.lower() in c.lower():
                return c
    return None

def _normalize_date(raw: str) -> str:
    import re as _re
    if not raw or raw.lower() in ("nan", "none", ""):
        return ""
    raw = raw.strip()
    has_year = bool(_re.search(r"\d{4}", raw))

    # 한국어 형식 직접 파싱
    m = _re.match(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", raw)
    if m:
        return f"{int(m.group(2)):02d}-{int(m.group(3)):02d}-{m.group(1)}"
    m = _re.match(r"(\d{1,2})월\s*(\d{1,2})일", raw)
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    # pandas 자동 추론 후 통일된 형식 출력
    try:
        dt = pd.to_datetime(raw, dayfirst=False)
        return dt.strftime("%m-%d-%Y") if has_year else dt.strftime("%m-%d")
    except Exception:
        pass

    # 마지막 fallback: 구분자만 통일
    return _re.sub(r"[./]", "-", raw)


def parse_csv(b: bytes) -> list[dict]:
    df = pd.read_csv(io.BytesIO(b), encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    cc = _detect(df.columns, CONTENT_ALIASES)
    if not cc:
        raise ValueError(f"내용 컬럼을 찾을 수 없어요. 현재 컬럼: {list(df.columns)}")
    rc = _detect(df.columns, RATING_ALIASES)
    dc = _detect(df.columns, DATE_ALIASES)
    ch = _detect(df.columns, CHANNEL_ALIASES)
    rows = []
    for i, row in df.iterrows():
        txt = str(row[cc]).strip()
        if not txt or txt.lower() == "nan":
            continue
        rows.append({
            "id":   i + 1,
            "내용": txt,
            "별점": row[rc]  if rc and pd.notna(row[rc])  else None,
            "날짜": _normalize_date(str(row[dc])) if dc and pd.notna(row[dc]) else "",
            "경로": str(row[ch]).strip() if ch and pd.notna(row[ch]) else "",
        })
    return rows

# ── 헬퍼 ─────────────────────────────────────────────────────────

def toggle_done(fb_id: int):
    if fb_id in st.session_state.completed:
        st.session_state.completed.discard(fb_id)
    else:
        st.session_state.completed.add(fb_id)

def _find_related(fb_id: int, enriched: list) -> list:
    """같은 채널 + 같은 리스크유형 + 유사 내용의 미완료 불만 목록 반환 (자신 제외)"""
    import re as _re
    target = next((f for f in enriched if f["id"] == fb_id), None)
    if not target:
        return []
    risk    = target.get("리스크유형", "")
    ch      = target.get("경로", "")
    words_t = set(_re.findall(r"[가-힣]{3,}", target.get("내용", "")))
    result  = []
    for fb in enriched:
        if fb["id"] == fb_id or fb["유형"] != "불만" or fb["id"] in st.session_state.completed:
            continue
        if ch and fb.get("경로", "") != ch:
            continue
        if fb.get("리스크유형", "") != risk:
            continue
        words_fb = set(_re.findall(r"[가-힣]{3,}", fb.get("내용", "")))
        if words_t & words_fb:
            result.append(fb)
    return result

@st.dialog("완료 처리 확인")
def _complete_dialog(fb_id: int, enriched: list):
    target  = next((f for f in enriched if f["id"] == fb_id), None)
    related = _find_related(fb_id, enriched)

    st.markdown(f"**완료 처리할 항목**")
    st.info(f"[{target.get('경로','—')}] {target['내용']}")

    if related:
        st.markdown(f"**같은 채널·같은 유형의 유사 불만 {len(related)}건**이 있어요. 함께 처리할까요?")
        for fb in related:
            st.write(f"• [{fb.get('경로','—')}] {fb['내용']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("모두 완료 처리", type="primary", use_container_width=True):
                st.session_state.completed.add(fb_id)
                for fb in related:
                    st.session_state.completed.add(fb["id"])
                st.rerun()
        with c2:
            if st.button("이 항목만 완료", use_container_width=True):
                st.session_state.completed.add(fb_id)
                st.rerun()
    else:
        if st.button("완료 처리", type="primary", use_container_width=True):
            st.session_state.completed.add(fb_id)
            st.rerun()

RANK_ICON  = ["🥇","🥈","🥉"]
RANK_CLASS = ["active-card-1","active-card-2","active-card-3"]

TYPE_COLOR = {"불만":"#FF4B4B","칭찬":"#2ECC71","문의":"#3498DB","요청":"#F39C12"}
TYPE_ICON  = {"불만":"😤","칭찬":"😊","문의":"🤔","요청":"💡"}

def urgency_dot(score: int) -> str:
    if score >= 15: return "🔴🔴"
    if score >= 10: return "🔴"
    if score >= 7:  return "🟠"
    if score >= 4:  return "🟡"
    if score >= 1:  return "⚪"
    return "—"

# ── 사이드바 ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")

    # Streamlit Cloud secrets에 저장된 키 자동 로드
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "Anthropic API Key",
        value=default_key,
        type="password",
        placeholder="sk-ant-...  (없으면 룰기반 분류)",
    )
    st.caption(
        "🔑 **본인의 Anthropic API Key를 입력하세요.**\n\n"
        "- 입력 시: Claude AI가 직접 분류 (더 정확)\n"
        "- 비워두면: 키워드 룰기반 분류 (무료)\n\n"
        "⚠️ 타인의 키를 입력하거나 공유하지 마세요. "
        "API 사용 비용은 키 소유자에게 청구됩니다.\n\n"
        "[키 발급 받기 →](https://console.anthropic.com/)"
    )

    st.divider()

    uploaded = st.file_uploader(
        "피드백 CSV 업로드",
        type=["csv"],
        help="'내용' 컬럼이 있으면 어떤 CSV든 OK",
    )

    if uploaded:
        file_id = uploaded.name + str(uploaded.size)
        if file_id != st.session_state.last_file:
            with st.spinner("분류 중…"):
                try:
                    feedbacks = parse_csv(uploaded.read())
                    enriched, mode = classify(feedbacks, api_key)
                    st.session_state.enriched  = enriched
                    st.session_state.mode      = mode
                    st.session_state.last_file = file_id
                    st.session_state.completed = set()   # 새 파일이면 완료 초기화
                except Exception as e:
                    st.error(f"오류: {e}")
                    st.stop()

    if st.session_state.enriched:
        completed = st.session_state.completed
        total     = len(st.session_state.enriched)
        done_cnt  = len(completed)
        st.divider()
        st.metric("전체", f"{total}건")
        st.metric("완료 처리됨", f"{done_cnt}건",
                  delta=f"미완료 {total - done_cnt}건 남음",
                  delta_color="inverse")
        mode_label = st.session_state.mode
        if "LLM" in mode_label:
            st.success(f"🤖 {mode_label}")
        else:
            st.warning(f"📐 {mode_label}")

# ── 데이터 없을 때 ────────────────────────────────────────────────

if st.session_state.enriched is None:
    st.title("☕ 피드백 모니터링 시스템")
    st.info("👈 왼쪽 사이드바에서 CSV를 업로드하면 자동으로 분류하고 대시보드를 보여드려요.")
    with st.expander("📌 지원 컬럼명 보기"):
        st.markdown("""
| 항목 | 인식하는 컬럼명 |
|---|---|
| **내용** (필수) | 내용, content, 피드백, feedback, comment, 텍스트, text |
| **별점** | 별점, rating, score, 점수, 평점, stars |
| **날짜** | 날짜, date, received_at, 받은날짜 |
| **경로** | 경로, channel, source, 채널 |
        """)
    st.stop()

# ── 대시보드 본문 ─────────────────────────────────────────────────

enriched  = st.session_state.enriched
completed = st.session_state.completed

st.title("☕ 고객 피드백 대시보드")
st.caption(f"총 {len(enriched)}건 · 완료 {len(completed)}건 / 미완료 {len(enriched)-len(completed)}건")
st.divider()

# ════════════════════════════════
# 섹션 1 — 급한 불만 Top3
# ════════════════════════════════

st.subheader("🚨 지금 가장 급한 불만 Top 3")

all_urgent = [fb for fb in enriched if fb["유형"] == "불만"]
all_urgent.sort(key=lambda x: (-x["긴급도"], x["id"]))
top3_active = [fb for fb in all_urgent if fb["id"] not in completed][:3]
top3_done   = [fb for fb in all_urgent if fb["id"] in completed]

if not top3_active and not top3_done:
    st.success("😊 긴급하게 대응할 불만이 없어요!")
else:
    # 활성 Top3
    if top3_active:
        for i, fb in enumerate(top3_active):
            tag   = RISK_CONFIG.get(fb.get("리스크유형",""), {}).get("label", "💬 불만")
            meta  = " · ".join(filter(None, [fb.get("날짜",""), fb.get("경로","")]))
            star  = f" · ★{fb['별점']}" if fb.get("별점") else ""
            score = fb.get("긴급도", 0)
            secondary = fb.get("2차피해유형", [])
            sec_badges = " ".join(
                f'<span style="background:#FF4B4B22;color:#FF4B4B;border-radius:10px;'
                f'padding:2px 8px;font-size:0.75rem;font-weight:700;">'
                f'{SECONDARY_CONFIG[s]["label"]}</span>'
                for s in secondary if s in SECONDARY_CONFIG
            )

            col_card, col_btn = st.columns([10, 2])
            with col_card:
                tag_html = f'<span style="background:{TYPE_COLOR["불만"]}25;color:{TYPE_COLOR["불만"]};border-radius:20px;padding:3px 11px;font-size:0.78rem;font-weight:700;">{tag}</span>'
                score_html = f'<span style="background:#88888822;color:#555;border-radius:20px;padding:3px 11px;font-size:0.78rem;font-weight:700;">점수 {score}</span>'
                meta_html = f'<div class="card-meta">{meta}{star}</div>' if meta else ''
                st.markdown(f"""<div class="{RANK_CLASS[i]}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span class="card-rank">{RANK_ICON[i]} {i+1}위</span>
    <span>{sec_badges} {tag_html} {score_html}</span>
  </div>
  <div class="card-quote">"{fb['내용']}"</div>
  <div class="card-reason">→ {fb.get('긴급이유','')}</div>
  {meta_html}
</div>""", unsafe_allow_html=True)
            with col_btn:
                st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
                if st.button("✅ 완료 처리", key=f"top3_done_{fb['id']}", use_container_width=True):
                    _complete_dialog(fb["id"], enriched)
                    st.rerun()
    else:
        st.success("🎉 급한 불만을 모두 처리했어요!")

    # 완료된 항목 (접어두기)
    if top3_done:
        with st.expander(f"✅ 처리 완료된 불만 {len(top3_done)}건"):
            for fb in top3_done:
                col_card, col_btn = st.columns([10, 2])
                with col_card:
                    st.markdown(f"""
<div class="done-card">
  <div class="card-rank">✅ 완료 처리됨</div>
  <div class="card-quote">"{fb['내용']}"</div>
</div>""", unsafe_allow_html=True)
                with col_btn:
                    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
                    if st.button("↩️ 되돌리기", key=f"top3_undo_{fb['id']}", use_container_width=True):
                        toggle_done(fb["id"])
                        st.rerun()

st.divider()

# ════════════════════════════════
# 섹션 2 — 유형별 개수
# ════════════════════════════════

st.subheader("📊 유형별 피드백 개수")

type_counts = get_type_counts(enriched)
type_order  = ["불만","칭찬","문의","요청"]

cols = st.columns(4)
for col, t in zip(cols, type_order):
    with col:
        c = type_counts.get(t, 0)
        col.metric(label=f"{TYPE_ICON[t]} {t}", value=f"{c}건")

fig = px.bar(
    x=type_order,
    y=[type_counts.get(t, 0) for t in type_order],
    color=type_order,
    color_discrete_map={t: TYPE_COLOR[t] for t in type_order},
    text=[type_counts.get(t, 0) for t in type_order],
    labels={"x":"유형","y":"건수"},
)
fig.update_traces(textposition="outside", marker_line_width=0)
fig.update_layout(
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(visible=False, showgrid=False),
    xaxis=dict(showgrid=False),
    margin=dict(t=30, b=10),
    height=260,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ════════════════════════════════
# 섹션 3 — 전체 피드백 테이블
# ════════════════════════════════

st.subheader(f"📋 전체 피드백 ({len(enriched)}건)")


# 필터
col_f1, col_f2, col_f3 = st.columns([2,2,2])
with col_f1:
    filter_type = st.multiselect("유형 필터", ["불만","칭찬","문의","요청"],
                                  default=["불만","칭찬","문의","요청"])
with col_f2:
    filter_status = st.radio("완료 상태", ["전체","미완료만","완료만"], horizontal=True)
with col_f3:
    sort_urgency = st.toggle("긴급도 높은 순 정렬", value=True)

rows = enriched[:]
if filter_type:
    rows = [r for r in rows if r["유형"] in filter_type]
if filter_status == "미완료만":
    rows = [r for r in rows if r["id"] not in completed]
elif filter_status == "완료만":
    rows = [r for r in rows if r["id"] in completed]
if sort_urgency:
    rows.sort(key=lambda x: (-x["긴급도"], x["id"]))

if not rows:
    st.info("조건에 맞는 피드백이 없어요.")
else:
    header = st.columns([0.4, 1, 1, 0.5, 0.8, 0.5, 4, 1.3])
    for col, label in zip(header, ["#","날짜","경로","별점","유형","긴급","내용","상태"]):
        col.markdown(f"**{label}**")
    st.markdown("<hr style='margin:4px 0 8px;border-color:#e0e0e0'>", unsafe_allow_html=True)

    for fb in rows:
        is_done = fb["id"] in completed
        row_cols = st.columns([0.4, 1, 1, 0.5, 0.8, 0.5, 4, 1.3])

        style = "color:#aaa;text-decoration:line-through;" if is_done else ""

        row_cols[0].markdown(f"<span style='{style}'>{fb['id']}</span>",    unsafe_allow_html=True)
        row_cols[1].markdown(f"<span style='{style}'>{fb.get('날짜','—')}</span>", unsafe_allow_html=True)
        row_cols[2].markdown(f"<span style='{style}'>{fb.get('경로','—')}</span>", unsafe_allow_html=True)
        row_cols[3].markdown(f"<span style='{style}'>{fb.get('별점','—')}</span>", unsafe_allow_html=True)

        tc = TYPE_COLOR.get(fb['유형'], '#888')
        row_cols[4].markdown(
            f"<span style='background:{tc}22;color:{tc};border-radius:10px;"
            f"padding:2px 9px;font-size:0.8rem;font-weight:700;{style}'>{fb['유형']}</span>",
            unsafe_allow_html=True)

        row_cols[5].markdown(
            f"<span style='{style}'>{urgency_dot(fb.get('긴급도', 0))}</span>",
            unsafe_allow_html=True)

        content_display = f"~~{fb['내용']}~~" if is_done else fb['내용']
        row_cols[6].markdown(f"<span style='{style}'>{fb['내용']}</span>", unsafe_allow_html=True)

        if is_done:
            if row_cols[7].button("↩️ 되돌리기", key=f"tbl_undo_{fb['id']}", use_container_width=True):
                toggle_done(fb["id"])
                st.rerun()
        else:
            if row_cols[7].button("✅ 완료 처리", key=f"tbl_done_{fb['id']}", use_container_width=True):
                _complete_dialog(fb["id"], enriched)
                st.rerun()