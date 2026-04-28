import streamlit as st
import uuid
import time
import datetime
import json
import io
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

st.set_page_config(page_title="ВЛК 2026: CDS Симулятор", layout="wide", page_icon="🏥")

# ==========================================
# УПРАВЛІННЯ СТАНОМ ТА БАЗА ЗНАНЬ
# ==========================================
def init_state():
    if 'step' not in st.session_state: st.session_state.step = 0
    if 'patient_id' not in st.session_state: st.session_state.patient_id = str(uuid.uuid4())[:8].upper()
    if 'patient_data' not in st.session_state: st.session_state.patient_data = {"icf_scores": {}}
    if 'paper_data' not in st.session_state: st.session_state.paper_data = {}
    if 'kep_signed' not in st.session_state: st.session_state.kep_signed = False
    if 'audit_log' not in st.session_state: st.session_state.audit_log = []
    if 'route' not in st.session_state: st.session_state.route = None

init_state()

def set_step(step):
    st.session_state.step = step

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()

def add_audit(action: str, detail: str = "", level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.audit_log.append({
        "ts": ts,
        "level": level,
        "action": action,
        "detail": detail
    })

# ==========================================
# БАЗА ЗНАНЬ
# ==========================================
KNOWLEDGE_BASE = {
    "Зір": {
        "icd": "H52.1 (Міопія)",
        "icf": "b210",
        "achi": "11212-00",
        "validation": "LOINC: 70914-7"
    },
    "Серце": {
        "icd": "I11.9 (Гіпертензивна хвороба)",
        "icf": "b420",
        "achi": "11700-00",
        "validation": "SNOMED: 38341003"
    },
    "Спина": {
        "icd": "M42.1 (Остеохондроз хребта)",
        "icf": "b710",
        "achi": "90901-03",
        "validation": "SNOMED: 282822008"
    },
    "Травлення": {
        "icd": "K29.3 (Хронічний гастрит)",
        "icf": "b515",
        "achi": "30473-00",
        "validation": "SNOMED: 8493009"
    },
    "Дихання": {
        "icd": "J45.9 (Астма, неуточнена)",
        "icf": "b440",
        "achi": "11503-05",
        "validation": "LOINC: 20150-9"
    },
    "Слух": {
        "icd": "H90.3 (Нейросенсорна туговухість)",
        "icf": "b230",
        "achi": "11309-00",
        "validation": "LOINC: 89020-2"
    }
}

SEVERITY_MAP = {
    "Легке порушення (.1)": 1,
    "Помірне порушення (.2)": 3,
    "Важке порушення (.3)": 10
}
OPTS = list(SEVERITY_MAP.keys())

# ==========================================
# MCDA (ВИПРАВЛЕНА ФОРМУЛА)
# ==========================================
def calculate_mcda_score(icf_scores):
    THRESHOLD = 10
    active = [v for v in icf_scores.values() if v > 0]
    if not active:
        return 0.0, "Придатний", 0, 0, 1.0

    M = max(active)
    S_rest = sum(active) - M
    alpha = max((THRESHOLD - M) / THRESHOLD, 0)  # БАГ ВИПРАВЛЕНО: alpha >= 0
    score = round(M + (S_rest * alpha), 2)

    if score < 3.0:
        status = "Придатний"
    elif score < 10.0:
        status = "Придатний до служби у військових частинах забезпечення, ТЦК та СП"
    else:
        status = "Непридатний"

    return score, status, M, S_rest, alpha

# ==========================================
# RADAR CHART
# ==========================================
def build_radar_chart(icf_scores: dict) -> bytes:
    domains = list(KNOWLEDGE_BASE.keys())
    values = [icf_scores.get(d, 0) for d in domains]
    N = len(domains)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, color="#c9d1d9", fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([1, 3, 10])
    ax.set_yticklabels(["1", "3", "10"], color="#8b949e", fontsize=7)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#30363d", linewidth=0.8)

    ax.plot(angles, values_plot, linewidth=2, linestyle="solid", color="#58a6ff")
    ax.fill(angles, values_plot, alpha=0.25, color="#58a6ff")

    # Threshold circle at 10
    circle_vals = [10] * N + [10]
    ax.plot(angles, circle_vals, linewidth=1, linestyle="dashed", color="#f85149", alpha=0.6)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ==========================================
# PDF ВИСНОВОК
# ==========================================
def build_pdf(pd_data: dict, score: float, status: str, M, S_rest, alpha,
              is_early_exit: bool, tdv_fail, audit_log: list) -> bytes:
    
    # 1. Завантаження та реєстрація кириличного шрифту
    font_path = "Roboto-Regular.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
        
    pdfmetrics.registerFont(TTFont('Roboto', font_path))

# ==========================================
# БОКОВЕ МЕНЮ
# ==========================================
with st.sidebar:
    st.header("📖 Пам'ятка для комісії")
    with st.expander("🛡️ Захист від фроду (MCDA)", expanded=True):
        st.write("MCDA-асимптота зупиняє накопичення балів. Статус 'Непридатний' заблоковано для легких порушень без раннього виходу.")
    with st.expander("⚡ Ранній вихід (Early Exit)", expanded=False):
        st.write("Важке порушення (.3) або ТДВ-конфлікт миттєво визначає статус 'Непридатний'.")
    with st.expander("🎯 ТДВ-конфлікт (Спецпосади)", expanded=False):
        st.write("Блокування призначення на цільову посаду при профільних відхиленнях (Снайпер → Зір, Водолаз → Слух/Дихання).")
    with st.expander("🔍 Аудит-лог", expanded=False):
        st.write("Кожна дія зберігається в лог з міткою часу. Лог включається в PDF-висновок.")
    st.markdown("---")
    st.caption("Версія: 2.0 | Повністю аудитована")

    # Аудит-лог у сайдбарі
    if st.session_state.audit_log:
        st.markdown("### 🔍 Аудит-лог")
        for entry in reversed(st.session_state.audit_log[-8:]):
            icon = "🔴" if entry["level"] == "ERROR" else "🟡" if entry["level"] == "WARN" else "🟢"
            st.caption(f"{icon} `{entry['ts']}` **{entry['action']}**")
            if entry["detail"]:
                st.caption(f"   ↳ {entry['detail']}")

st.progress((st.session_state.step + 1) / 4, text=f"Етап: {st.session_state.step + 1} / 4")
st.write("---")

# ==========================================
# КРОК 0: ПРОФІЛЬ ПАЦІЄНТА
# ==========================================
if st.session_state.step == 0:
    st.title("⚙️ Адмін-панель: Профіль пацієнта")
    col1, col2 = st.columns([1, 2])
    with col1:
        pib = st.text_input("ПІБ Пацієнта:", "Коваленко Іван Петрович")
        st.code(f"ID в ЕСОЗ: {st.session_state.patient_id}")
        prof = st.selectbox("Військова посада (ТДВ):", ["Снайпер", "Водолаз", "Піхотинець (Загальні)"])
        has_unstructured = st.checkbox("Симулювати неструктурований запис", value=False)
    with col2:
        st.write("Відмітьте системи, у яких є зафіксовані порушення:")
        active_scores = {}
        for domain, db in KNOWLEDGE_BASE.items():
            if st.checkbox(f"Включити домен: {domain} ({db['icd']})", key=f"chk_{domain}"):
                val = st.selectbox(f"Тяжкість ({db['icf']}):", OPTS, key=f"sel_{domain}")
                active_scores[domain] = SEVERITY_MAP[val]

    if st.button("Зберегти ➔", type="primary"):
        if not active_scores:
            st.error("Оберіть хоча б одне порушення.")
        else:
            st.session_state.patient_data = {
                "pib": pib,
                "id": st.session_state.patient_id,
                "prof": prof,
                "unstructured": has_unstructured,
                "icf_scores": active_scores
            }
            add_audit("Профіль створено", f"Пацієнт: {pib} | Посада: {prof} | Доменів: {len(active_scores)}")
            set_step(1)
            st.rerun()

# ==========================================
# КРОК 1: ВИБІР МАРШРУТУ
# ==========================================
elif st.session_state.step == 1:
    pd = st.session_state.patient_data
    st.title("🏥 Вибір маршруту")
    st.success(f"**Пацієнт:** {pd['pib']} | **ID:** {pd['id']} | **Посада (ТДВ):** {pd['prof']}")
    route = st.radio("Рівень автоматизації CDS:", ["1) Автоматичний", "2) Гібридний", "3) Паперовий (Legacy)"])

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Назад"):
            set_step(0); st.rerun()
    with col_next:
        if st.button("Запустити ➔", type="primary"):
            r = 1 if "Автоматичний" in route else 2 if "Гібридний" in route else 3
            st.session_state.route = r
            add_audit("Маршрут обрано", f"{'Автоматичний' if r==1 else 'Гібридний' if r==2 else 'Паперовий'}")
            set_step(2); st.rerun()

# ==========================================
# КРОК 2: МАПІНГ ТА ВАЛІДАЦІЯ
# ==========================================
elif st.session_state.step == 2:
    pd = st.session_state.patient_data

    if st.session_state.route in [1, 2]:
        st.title("🧠 CDS Мапінг")
        with st.spinner("Синхронізація ЕСОЗ..."):
            time.sleep(0.8)

        # Build tree HTML for each domain
        domains_data = []
        for domain in pd["icf_scores"].keys():
            db = KNOWLEDGE_BASE[domain]
            sev_val = pd["icf_scores"][domain]
            severity_text = [k for k, v in SEVERITY_MAP.items() if v == sev_val][0].split(" ")[-1]
            sev_color = "#f85149" if sev_val == 10 else "#d29922" if sev_val == 3 else "#3fb950"
            domains_data.append({
                "domain": domain,
                "icd": db["icd"],
                "icf": db["icf"],
                "sev": severity_text,
                "sev_color": sev_color,
                "achi": db["achi"],
                "validation": db["validation"],
            })

        # Generate tree cards HTML
        cards_html = ""
        for i, d in enumerate(domains_data):
            delay_base = i * 0.18
            cards_html += f"""
            <div class="tree-card" style="animation-delay:{delay_base:.2f}s">
              <div class="card-title">{d['domain']}</div>

              <div class="node icd-node" style="animation-delay:{delay_base+0.05:.2f}s">
                <span class="node-label">ICD-10</span>
                <span class="node-value">{d['icd']}</span>
              </div>

              <div class="connector-wrap">
                <div class="connector-line" style="animation-delay:{delay_base+0.15:.2f}s"></div>
                <div class="connector-arrow" style="animation-delay:{delay_base+0.2:.2f}s">▼</div>
                <div class="connector-tag" style="animation-delay:{delay_base+0.18:.2f}s">TRIGGERS</div>
              </div>

              <div class="node icf-node" style="animation-delay:{delay_base+0.25:.2f}s">
                <span class="node-label">ICF Core</span>
                <span class="node-value">{d['icf']}</span>
                <span class="sev-badge" style="background:{d['sev_color']}22;color:{d['sev_color']};border-color:{d['sev_color']}55">{d['sev']}</span>
              </div>

              <div class="connector-wrap">
                <div class="connector-line" style="animation-delay:{delay_base+0.32:.2f}s"></div>
                <div class="connector-arrow" style="animation-delay:{delay_base+0.37:.2f}s">▼</div>
                <div class="connector-tag" style="animation-delay:{delay_base+0.35:.2f}s">TRIGGERS</div>
              </div>

              <div class="node achi-node" style="animation-delay:{delay_base+0.42:.2f}s">
                <span class="node-label">ACHI</span>
                <span class="node-value">{d['achi']}</span>
              </div>

              <div class="connector-wrap">
                <div class="connector-line" style="animation-delay:{delay_base+0.5:.2f}s"></div>
                <div class="connector-arrow" style="animation-delay:{delay_base+0.55:.2f}s">▼</div>
                <div class="connector-tag" style="animation-delay:{delay_base+0.52:.2f}s">TRIGGERS</div>
              </div>

              <div class="node val-node" style="animation-delay:{delay_base+0.6:.2f}s">
                <span class="node-label">VALIDATION</span>
                <span class="node-value">{d['validation']}</span>
              </div>

              <div class="card-ok" style="animation-delay:{delay_base+0.75:.2f}s">✓ MAPPED</div>
            </div>
            """

        html_tree = f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

          .tree-root {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            padding: 8px 0 20px 0;
            font-family: 'JetBrains Mono', monospace;
          }}

          .tree-card {{
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 16px 14px 12px 14px;
            min-width: 190px;
            flex: 1 1 190px;
            max-width: 240px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0;
            opacity: 0;
            transform: translateY(18px);
            animation: fadeUp 0.45s cubic-bezier(.22,1,.36,1) forwards;
            box-shadow: 0 0 0 0 #58a6ff00;
            transition: box-shadow 0.3s, border-color 0.3s;
          }}
          .tree-card:hover {{
            border-color: #388bfd55;
            box-shadow: 0 0 18px 2px #388bfd18;
          }}

          @keyframes fadeUp {{
            to {{ opacity:1; transform:translateY(0); }}
          }}

          .card-title {{
            font-family: 'Syne', sans-serif;
            font-size: 13px;
            font-weight: 800;
            color: #e6edf3;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 12px;
          }}

          .node {{
            width: 100%;
            border-radius: 7px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 3px;
            opacity: 0;
            animation: fadeUp 0.35s cubic-bezier(.22,1,.36,1) forwards;
          }}
          .node-label {{
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }}
          .node-value {{
            font-size: 11px;
            font-weight: 600;
            word-break: break-all;
          }}

          .icd-node  {{ background:#161b22; border:1px solid #30363d; }}
          .icd-node .node-label  {{ color:#58a6ff; }}
          .icd-node .node-value  {{ color:#c9d1d9; }}

          .icf-node  {{ background:#161b22; border:1px solid #30363d; }}
          .icf-node .node-label  {{ color:#bc8cff; }}
          .icf-node .node-value  {{ color:#c9d1d9; }}

          .achi-node {{ background:#161b22; border:1px solid #30363d; }}
          .achi-node .node-label {{ color:#ffa657; }}
          .achi-node .node-value {{ color:#c9d1d9; }}

          .val-node  {{ background:#161b22; border:1px solid #30363d; }}
          .val-node .node-label  {{ color:#3fb950; }}
          .val-node .node-value  {{ color:#c9d1d9; }}

          .sev-badge {{
            display:inline-block;
            font-size: 8px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid;
            margin-top: 2px;
            letter-spacing: 0.05em;
            width: fit-content;
          }}

          .connector-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            width: 100%;
            gap: 0;
          }}
          .connector-line {{
            width: 1px;
            height: 10px;
            background: linear-gradient(to bottom, #30363d, #58a6ff44);
            opacity: 0;
            animation: fadeUp 0.2s ease forwards;
          }}
          .connector-arrow {{
            font-size: 8px;
            color: #58a6ff;
            opacity: 0;
            animation: fadeUp 0.2s ease forwards;
            line-height: 1;
          }}
          .connector-tag {{
            font-size: 7px;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #8b949e;
            opacity: 0;
            animation: fadeUp 0.2s ease forwards;
            margin: 1px 0 3px 0;
          }}

          .card-ok {{
            margin-top: 10px;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: #3fb950;
            opacity: 0;
            animation: fadeUp 0.3s ease forwards;
          }}
        </style>

        <div class="tree-root">
          {cards_html}
        </div>
        """

        import streamlit.components.v1 as components
        components.html(html_tree, height=460, scrolling=True)

        add_audit("Маппінг виконано", f"Доменів: {len(pd['icf_scores'])}", "INFO")

        if pd["unstructured"] and st.session_state.route == 1:
            st.error("❌ АВТО-МАРШРУТ ПЕРЕРВАНО: Потрібен КЕП для неструктурованих даних.")
            add_audit("Маршрут перервано", "Неструктуровані дані без КЕП", "ERROR")
            if st.button("⬅️ Назад"):
                set_step(1); st.rerun()

        elif pd["unstructured"] and st.session_state.route == 2:
            if not st.session_state.kep_signed:
                if st.button("✍️ Валідувати КЕП"):
                    st.session_state.kep_signed = True
                    add_audit("КЕП накладено", "Гібридний маршрут", "INFO")
                    st.rerun()
            else:
                st.success("✅ КЕП накладено.")
                if st.button("Далі ➔", type="primary"):
                    set_step(3); st.rerun()
        else:
            col_back, col_next = st.columns([1, 5])
            with col_back:
                if st.button("⬅️ Назад"):
                    set_step(1); st.rerun()
            with col_next:
                if st.button("Далі ➔", type="primary"):
                    set_step(3); st.rerun()

    elif st.session_state.route == 3:
        st.title("📝 Ручний ввід (Паперовий маршрут)")
        manual_icd = st.selectbox("МКХ-10:", [db["icd"] for db in KNOWLEDGE_BASE.values()])
        target_domain = next(dom for dom, db in KNOWLEDGE_BASE.items() if db["icd"] == manual_icd)
        manual_icf = st.selectbox("МКФ:", list(set(db["icf"] for db in KNOWLEDGE_BASE.values())))
        manual_achi = st.selectbox("ACHI:", list(set(db["achi"] for db in KNOWLEDGE_BASE.values())))
        manual_sev = st.selectbox("Тяжкість:", OPTS)

        if st.button("Додати ➕"):
            if manual_icf != KNOWLEDGE_BASE[target_domain]["icf"]:
                st.error("❌ ПОМИЛКА: Невірний зв'язок МКХ та МКФ.")
                add_audit("Помилка валідації", f"МКХ {manual_icd} ↛ МКФ {manual_icf}", "ERROR")
            else:
                st.session_state.paper_data[target_domain] = SEVERITY_MAP[manual_sev]
                add_audit("Запис додано", f"{target_domain}: {manual_sev}", "INFO")
                st.success("✅ Додано.")

        if st.session_state.paper_data:
            st.write("**Введені дані:**", st.session_state.paper_data)

        col_back, col_next = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Назад"):
                set_step(1); st.rerun()
        with col_next:
            if st.button("Далі ➔", type="primary"):
                if not st.session_state.paper_data:
                    st.error("Додайте хоча б один запис.")
                else:
                    st.session_state.patient_data["icf_scores"] = st.session_state.paper_data
                    set_step(3); st.rerun()

# ==========================================
# КРОК 3: ФІНАЛЬНИЙ СТАТУС
# ==========================================
elif st.session_state.step == 3:
    pd = st.session_state.patient_data
    st.title("📋 Фінальний Висновок")

    is_early_exit = any(s == 10 for s in pd["icf_scores"].values())
    tdv_fail = None
    if pd["prof"] == "Снайпер" and pd["icf_scores"].get("Зір", 0) > 0:
        tdv_fail = "Снайпер: Порушення зору."
    elif pd["prof"] == "Водолаз" and (pd["icf_scores"].get("Слух", 0) > 0 or pd["icf_scores"].get("Дихання", 0) > 0):
        tdv_fail = "Водолаз: Порушення слуху або дихання."

    score, status, M, S_rest, alpha = calculate_mcda_score(pd["icf_scores"])

    # --- Radar chart ---
    col_chart, col_verdict = st.columns([1, 1])
    with col_chart:
        st.subheader("📊 Radar: ICF Профіль")
        chart_bytes = build_radar_chart(pd["icf_scores"])
        st.image(chart_bytes, use_container_width=True)

    with col_verdict:
        st.subheader("⚖️ Статус")
        if is_early_exit or tdv_fail:
            st.error("🚨 НЕПРИДАТНИЙ (Ранній вихід)")
            if is_early_exit:
                st.info("Базова причина: Знайдено кваліфікатор .3 (Важке порушення).")
                add_audit("Early Exit", "Кваліфікатор .3 знайдено", "WARN")
            if tdv_fail:
                st.warning(f"ТДВ причина: {tdv_fail}")
                add_audit("ТДВ-конфлікт", tdv_fail, "WARN")
        else:
            if status == "Придатний":
                st.success(f"✅ **{status}**")
            else:
                st.warning(f"⚠️ **{status}**")
            add_audit("MCDA завершено", f"Бал: {score} | Статус: {status}", "INFO")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("M (домінант)", M)
        c2.metric("S_rest (фон)", S_rest)
        c3.metric("α (запас)", f"{int(alpha*100)}%")
        st.metric("Підсумковий бал MCDA", score)

    st.markdown("---")

    # --- Аудит-лог таблиця ---
    st.subheader("🔍 Аудит-лог сесії")
    if st.session_state.audit_log:
        log_data = []
        for e in st.session_state.audit_log:
            icon = "🔴" if e["level"] == "ERROR" else "🟡" if e["level"] == "WARN" else "🟢"
            log_data.append({"Час": e["ts"], "Рівень": f"{icon} {e['level']}", "Дія": e["action"], "Деталь": e["detail"]})
        st.table(log_data)
    else:
        st.info("Лог порожній.")

    st.markdown("---")

    # --- PDF Export ---
    st.subheader("📄 Завантажити висновок")
    pdf_bytes = build_pdf(pd, score, status, M, S_rest, alpha, is_early_exit, tdv_fail, st.session_state.audit_log)
    filename = f"VLK_{pd.get('id','')}_висновок.pdf"
    st.download_button(
        label="⬇️ Завантажити PDF-висновок",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        type="primary"
    )

    st.markdown("---")
    if st.button("🔄 Почати знову"):
        reset_all(); st.rerun()
