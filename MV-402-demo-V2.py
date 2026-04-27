import streamlit as st
import uuid
import time
import datetime
import json
import io

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
        "achi": "11212-00 (Дослідження очного дна)",
        "validation": "LOINC: 70914-7 (Гострота зору)"
    },
    "Серце": {
        "icd": "I11.9 (Гіпертензивна хвороба)",
        "icf": "b420",
        "achi": "11700-00 (ЕКГ, 12 відведень)",
        "validation": "SNOMED: 38341003 (Гіпертензивна хвороба серця)"
    },
    "Спина": {
        "icd": "M42.1 (Остеохондроз хребта)",
        "icf": "b710",
        "achi": "90901-03 (МРТ хребта)",
        "validation": "SNOMED: 282822008 (Протрузія диска)"
    },
    "Травлення": {
        "icd": "K29.3 (Хронічний гастрит)",
        "icf": "b515",
        "achi": "30473-00 (Панендоскопія)",
        "validation": "SNOMED: 8493009 (Хронічний гастрит)"
    },
    "Дихання": {
        "icd": "J45.9 (Астма, неуточнена)",
        "icf": "b440",
        "achi": "11503-05 (Спірометрія)",
        "validation": "LOINC: 20150-9 (ОФВ1/FEV1)"
    },
    "Слух": {
        "icd": "H90.3 (Нейросенсорна туговухість)",
        "icf": "b230",
        "achi": "11309-00 (Аудіометрія)",
        "validation": "LOINC: 89020-2 (Поріг чутності)"
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
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("vtitle", parent=styles["Title"], fontSize=14, spaceAfter=4)
    style_sub   = ParagraphStyle("vsub",   parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    style_h2    = ParagraphStyle("vh2",    parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4)
    style_body  = ParagraphStyle("vbody",  parent=styles["Normal"], fontSize=9, leading=13)
    style_stamp = ParagraphStyle("vstamp", parent=styles["Normal"], fontSize=11, textColor=colors.red,
                                 alignment=TA_CENTER, spaceAfter=6)

    story = []

    # Заголовок
    story.append(Paragraph("ВІЙСЬКОВО-ЛІКАРСЬКА КОМІСІЯ", style_title))
    story.append(Paragraph("Висновок CDS Rule Engine | Версія 2.0", style_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#444444")))
    story.append(Spacer(1, 0.3*cm))

    # Дані пацієнта
    story.append(Paragraph("1. Дані пацієнта", style_h2))
    pt_data = [
        ["ПІБ:", pd_data.get("pib", "—")],
        ["ID в ЕСОЗ:", pd_data.get("id", "—")],
        ["Посада (ТДВ):", pd_data.get("prof", "—")],
        ["Дата висновку:", datetime.datetime.now().strftime("%d.%m.%Y %H:%M")],
    ]
    t = Table(pt_data, colWidths=[4*cm, 12*cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#555555")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # Домени
    story.append(Paragraph("2. Функціональні домени (ICF-маппінг)", style_h2))
    domain_rows = [["Домен", "МКХ-10", "ICF", "ACHI", "Тяжкість"]]
    for domain, score_val in pd_data.get("icf_scores", {}).items():
        db = KNOWLEDGE_BASE.get(domain, {})
        sev_label = [k for k, v in SEVERITY_MAP.items() if v == score_val]
        sev_text = sev_label[0].split(" ")[-1] if sev_label else str(score_val)
        domain_rows.append([domain, db.get("icd",""), db.get("icf",""), db.get("achi",""), sev_text])
    dt = Table(domain_rows, colWidths=[2.5*cm, 4*cm, 2*cm, 5*cm, 2.5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d333b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#444444")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f8f8"), colors.white]),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.3*cm))

    # MCDA
    story.append(Paragraph("3. MCDA Розрахунок", style_h2))
    mcda_rows = [
        ["Домінуючий тягар (M)", str(M)],
        ["Фоновий тягар (S_rest)", str(S_rest)],
        ["Запас міцності (α)", f"{int(alpha*100)}%"],
        ["Підсумковий бал", str(score)],
    ]
    mt = Table(mcda_rows, colWidths=[8*cm, 8*cm])
    mt.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#555555")),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.4*cm))

    # Статус (штамп)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#444444")))
    story.append(Spacer(1, 0.3*cm))
    if is_early_exit or tdv_fail:
        story.append(Paragraph("◼ НЕПРИДАТНИЙ (РАННІЙ ВИХІД)", style_stamp))
        if is_early_exit:
            story.append(Paragraph("Причина: Знайдено кваліфікатор .3 (Важке порушення)", style_body))
        if tdv_fail:
            story.append(Paragraph(f"ТДВ причина: {tdv_fail}", style_body))
    else:
        color_status = colors.green if "Придатний" == status else colors.orange
        st_style = ParagraphStyle("vstatus", parent=style_stamp, textColor=color_status)
        story.append(Paragraph(f"◼ {status.upper()}", st_style))

    story.append(Spacer(1, 0.5*cm))

    # Аудит-лог
    story.append(Paragraph("4. Аудит-лог сесії", style_h2))
    audit_rows = [["Час", "Рівень", "Дія", "Деталь"]]
    for entry in audit_log:
        audit_rows.append([entry["ts"], entry["level"], entry["action"], entry["detail"]])
    at = Table(audit_rows, colWidths=[1.5*cm, 1.5*cm, 5*cm, 8*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d333b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f8f8"), colors.white]),
    ]))
    story.append(at)

    doc.build(story)
    buf.seek(0)
    return buf.read()

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
            time.sleep(0.5)

        cols = st.columns(len(pd["icf_scores"]))
        for idx, domain in enumerate(pd["icf_scores"].keys()):
            db = KNOWLEDGE_BASE[domain]
            sev_val = pd["icf_scores"][domain]
            severity_text = [k for k, v in SEVERITY_MAP.items() if v == sev_val][0].split(" ")[-1]
            with cols[idx]:
                st.code(
                    f"[1. Діагноз] МКХ-10: {db['icd']}\n"
                    f"[2. Функція] ICF: {db['icf']} ({severity_text})\n"
                    f"[3. Доказ] ACHI: {db['achi']}\n"
                    f"[4. Валідація] {db['validation']}",
                    language="yaml"
                )

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