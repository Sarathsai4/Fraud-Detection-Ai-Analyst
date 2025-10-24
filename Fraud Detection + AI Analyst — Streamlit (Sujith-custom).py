"""
Author: Sarath Sai Sujith Srinivas Grandhe
Title: Fraud Detection + AI Analysis — Streamlit (Sujith-custom)

Notes:
- Original implementation written from scratch. Structure/names chosen to reflect a distinct, personal coding style.
- Adds: time-aware CV, cost-sensitive thresholding, AI Analyst (LLM) that interprets metrics & answers questions.

Run:
  pip install streamlit scikit-learn lightgbm matplotlib pandas numpy requests pydantic
  streamlit run "Fraud Detection + AI Analyst — Streamlit (Sujith-custom).py"

"""

from __future__ import annotations

import os
import io
import json
import base64
import sys
import site
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ---------------- PDF export helper (ReportLab, auto-install if needed) ----------------
import importlib.util, subprocess
if importlib.util.find_spec("reportlab") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "reportlab"])

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

def _make_pdf_bytes(title: str, body_text: str) -> bytes:
    """Create a clean, printable PDF from plain text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 56  # ~0.78in
    x, y = margin, height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 24

    # Body
    c.setFont("Helvetica", 10)
    line_gap = 13
    para_gap = 6
    max_width = width - 2 * margin

    for line in body_text.splitlines():
        wrapped = simpleSplit(line, "Helvetica", 10, max_width)
        for w in wrapped:
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(x, y, w)
            y -= line_gap
        y -= para_gap

    c.save()
    buf.seek(0)
    return buf.getvalue()
# --------------------------------------------------------------------------------------

# LightGBM import (safe path fix)
try:
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    for p in user_sites:
        if p and p not in sys.path:
            sys.path.append(p)
except Exception:
    pass

try:
    import lightgbm as _lgbm
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
    _LGBM_ERR = ""
except Exception as e:
    _HAS_LGBM = False
    _LGBM_ERR = repr(e)


# ------------------------------
# Configuration & Data Classes
# ------------------------------
@dataclass
class SaModelCfg:
    algo: str = "lgbm"
    n_splits: int = 5
    random_state: int = 42
    time_aware: bool = False
    time_col: Optional[str] = None


@dataclass
class SaAiCfg:
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ------------------------------
# Helpers
# ------------------------------
def sa_safe_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def sa_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    drop_like = {"Unnamed: 0", "index"}
    return df[[c for c in df.columns if c not in drop_like]]


def sa_encode_cats(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[OrdinalEncoder]]:
    exclude = set(exclude or [])
    cat_cols = [c for c in df.columns if df[c].dtype == "O" and c not in exclude]
    if not cat_cols:
        return df, None
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    return df, enc


def sa_download_csv(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">Download {filename}</a>'


# ------------------------------
# Target column normalization
# ------------------------------
def sa_try_map_target(df: pd.DataFrame) -> pd.DataFrame:
    if "isFraud" not in df.columns:
        cand_lower = {c.lower(): c for c in df.columns}
        candidates = ["isfraud", "class", "fraud", "target", "label"]
        found = next((cand_lower[c] for c in candidates if c in cand_lower), None)
        if found:
            df.rename(columns={found: "isFraud"}, inplace=True)
        else:
            st.error("The dataset must contain a binary target column named 'isFraud' (or Class/Fraud/Target/Label).")
            st.stop()

    col = "isFraud"
    if df[col].dtype == "O":
        mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
    if not np.issubdtype(df[col].dtype, np.number):
        df[col] = df[col].astype(int)
    return df


# ------------------------------
# CV & Model Training
# ------------------------------
class SaTimeBlockedKFold:
    def __init__(self, n_splits: int, time_col: str):
        self.n_splits = n_splits
        self.time_col = time_col

    def split(self, X: pd.DataFrame, y: pd.Series):
        order = np.argsort(X[self.time_col].values)
        fold_sizes = np.full(self.n_splits, len(X) // self.n_splits, dtype=int)
        fold_sizes[: len(X) % self.n_splits] += 1
        current = 0
        folds = []
        for fs in fold_sizes:
            folds.append(order[current: current + fs])
            current += fs
        for i in range(self.n_splits):
            va_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield tr_idx, va_idx


def sa_build_model(cfg: SaModelCfg):
    if cfg.algo == "lgbm" and _HAS_LGBM:
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="binary",
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    return LogisticRegression(max_iter=300)


@st.cache_data(show_spinner=False)
def sa_kfold_oof(X: pd.DataFrame, y: pd.Series, cfg: SaModelCfg) -> Tuple[np.ndarray, Optional[pd.Series]]:
    splitter = (
        SaTimeBlockedKFold(cfg.n_splits, cfg.time_col)
        if cfg.time_aware and cfg.time_col and cfg.time_col in X.columns
        else StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    )
    oof = np.zeros(len(X))
    importances: List[pd.Series] = []
    # Allow both splitter types
    splits = splitter.split(X, y) if hasattr(splitter, "split") else splitter.split(X, y)
    for tr_idx, va_idx in splits:
        model = sa_build_model(cfg)
        X_tr, X_va, y_tr, y_va = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        if _HAS_LGBM and isinstance(model, LGBMClassifier):
            importances.append(pd.Series(model.feature_importances_, index=X.columns))
    feat_imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances else None
    return oof, feat_imp


# ------------------------------
# Metrics
# ------------------------------
def sa_counts_at_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    acc  = (tp + tn) / len(y_true)
    fpr  = fp / (fp + tn + 1e-9)
    return {"threshold": thr, "precision": prec, "recall": rec, "accuracy": acc, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def sa_threshold_table(y_true, y_prob, grid):
    return pd.DataFrame([sa_counts_at_threshold(y_true, y_prob, float(t)) for t in grid])


def sa_cost_of_threshold(row, cost_fp, cost_fn):
    return row["fp"] * cost_fp + row["fn"] * cost_fn


# ------------------------------
# AI Client
# ------------------------------
def sa_call_llm(system_prompt: str, user_prompt: str, ai_cfg: SaAiCfg) -> str:
    if not ai_cfg.api_key:
        return "(AI disabled — provide API key.)"
    import requests
    headers = {"Authorization": f"Bearer {ai_cfg.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": ai_cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    url = ai_cfg.base_url.rstrip("/") + "/v1/chat/completions"
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(AI error: {e})"


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Fraud Detection + AI Analysis", layout="wide")
st.title("💳 Fraud Detection + AI Analyst ")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    upload_mode = st.radio("Format", ["Merged CSV (has isFraud)", "Transaction.csv + Identity.csv"])
    merged_file = trans_file = id_file = None
    if upload_mode == "Merged CSV (has isFraud)":
        merged_file = st.file_uploader("Upload merged CSV", type=["csv"])
    else:
        trans_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
        id_file = st.file_uploader("Upload Identity CSV", type=["csv"])

    st.divider()
    algo_key = "lgbm" if st.selectbox("Algorithm", ["LightGBM", "LogisticRegression"]) == "LightGBM" else "logreg"
    n_splits = st.slider("K-Folds", 3, 10, 5)
    seed = st.number_input("Random seed", 0, 99999, 42)
    time_aware = st.toggle("Time-aware CV", False)
    time_col = st.text_input("Time column") if time_aware else None
    st.divider()
    cost_fp = st.number_input("Cost of False Positive", 0.0, step=0.1, value=1.0)
    cost_fn = st.number_input("Cost of False Negative", 0.0, step=0.5, value=10.0)

    st.divider()
    base_url = st.text_input("LLM API Base URL", "https://api.groq.com/openai")
    api_key = st.text_input("API Key", type="password")
    model = st.text_input("Model name", "groq/compound")

# Load data
_df = None
if upload_mode == "Merged CSV (has isFraud)" and merged_file:
    _df = sa_safe_csv(merged_file)
elif trans_file and id_file:
    tdf, idf = sa_basic_clean(sa_safe_csv(trans_file)), sa_basic_clean(sa_safe_csv(id_file))
    if "TransactionID" in tdf.columns and "TransactionID" in idf.columns:
        _df = tdf.merge(idf, on="TransactionID", how="left")
if _df is None:
    st.info("Upload data to continue.")
    st.stop()

_df = sa_try_map_target(_df)
st.subheader("Preview")
st.dataframe(_df.head())

y = _df["isFraud"].astype(int)
X = sa_basic_clean(_df.drop(columns=["isFraud"]))
if time_aware and time_col and time_col in X.columns:
    X[time_col] = pd.to_datetime(X[time_col], errors="coerce").astype("int64") // 10**9
X_enc, _ = sa_encode_cats(X, exclude=[time_col] if time_col else None)

cfg = SaModelCfg(algo=algo_key, n_splits=n_splits, random_state=seed, time_aware=time_aware, time_col=time_col)
with st.spinner("Training..."):
    oof_prob, feat_imp = sa_kfold_oof(X_enc, y, cfg)
    auc = roc_auc_score(y, oof_prob)

# Theme
theme_colors = dict(primary="#1f77b4", secondary="#ff7f0e")

# ROC Curve
fpr, tpr, _ = roc_curve(y, oof_prob)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.4f}", line=dict(color=theme_colors["primary"], width=3)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="#aaa")))
fig_roc.update_layout(title="ROC Curve (OOF)", template="plotly_white", title_x=0.5)
buf_roc = io.BytesIO(); fig_roc.write_image(buf_roc, format="png")
st.download_button("📈 Download ROC (PNG)", data=buf_roc.getvalue(), file_name="roc_curve.png", mime="image/png")
st.plotly_chart(fig_roc, use_container_width=True)

# PR Curve
prec, rec, _ = precision_recall_curve(y, oof_prob)
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=theme_colors["secondary"], width=3)))
fig_pr.update_layout(title="Precision-Recall Curve", template="plotly_white", title_x=0.5)
buf_pr = io.BytesIO(); fig_pr.write_image(buf_pr, format="png")
st.download_button("📊 Download PR (PNG)", data=buf_pr.getvalue(), file_name="precision_recall.png", mime="image/png")
st.plotly_chart(fig_pr, use_container_width=True)

# Threshold Explorer
st.subheader("Threshold Explorer")
thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
stats = sa_counts_at_threshold(y, oof_prob, thr)
cm = np.array([[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]])
fig_cm = ff.create_annotated_heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues")
fig_cm.update_layout(title=f"Confusion Matrix (Thr={thr:.2f})", template="plotly_white", title_x=0.5)
buf_cm = io.BytesIO(); fig_cm.write_image(buf_cm, format="png")
st.download_button("🧮 Download CM (PNG)", data=buf_cm.getvalue(), file_name="confusion_matrix.png", mime="image/png")
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown(
    f"**Precision:** {stats['precision']:.3f} **Recall:** {stats['recall']:.3f} "
    f"**Accuracy:** {stats['accuracy']:.3f} **FPR:** {stats['fpr']:.3f}"
)

grid = np.linspace(0.05, 0.95, 19)
table = sa_threshold_table(y, oof_prob, grid)
st.dataframe(table.style.format("{:.3f}"))
st.markdown(sa_download_csv(table, "threshold_metrics.csv"), unsafe_allow_html=True)

if cost_fp > 0 or cost_fn > 0:
    costs = [sa_cost_of_threshold(row, cost_fp, cost_fn) for _, row in table.iterrows()]
    best_thr = float(table.iloc[int(np.argmin(costs))]["threshold"])
    st.info(f"💰 Cost-aware optimal threshold: **{best_thr:.2f}**")

# Feature Importance
if feat_imp is not None and len(feat_imp) > 0:
    st.subheader("Feature Importance")
    top_k = st.slider("Top features", 5, min(50, len(feat_imp)), min(20, len(feat_imp)))
    imp_df = feat_imp.head(top_k).iloc[::-1].reset_index()
    imp_df.columns = ["Feature", "Importance"]
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[theme_colors["primary"]])
    fig_imp.update_layout(title="Feature Importance", template="plotly_white", title_x=0.5)
    buf_imp = io.BytesIO(); fig_imp.write_image(buf_imp, format="png")
    st.download_button("🏗️ Download Feature Importance (PNG)", data=buf_imp.getvalue(), file_name="feature_importance.png", mime="image/png")
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------- AI Analyst — with NumPy-safe JSON + PDF ----------------------
st.divider()
st.header("🤖 AI Analyst — Summary & Q&A")
ai_cfg = SaAiCfg(base_url=base_url, api_key=api_key, model=model)
ai_system = (
    "You are a data science copilot. Interpret model metrics (ROC/PR), threshold trade-offs, and "
    "cost-sensitive recommendations. Provide concise, actionable insights for fraud detection optimization."
)

def sa_numpy_converter(o):
    # Convert numpy types to native Python for safe JSON serialization
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

ai_summary = {
    "auc": float(auc),
    "threshold": float(thr),
    "metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in stats.items()},
    "pos_rate": float(y.mean()),
    "costs": {"fp": float(cost_fp), "fn": float(cost_fn)},
    "algo": cfg.algo,
}

with st.expander("Show JSON Summary"):
    st.json(ai_summary)

summary_text = json.dumps(ai_summary, indent=2, default=sa_numpy_converter)
auto_brief = sa_call_llm(ai_system, f"Analyze this fraud detection summary:\n{summary_text}", ai_cfg)

st.subheader("AI — Recommendations")
st.write(auto_brief)

# --- New: PDF download button for the AI summary ---
pdf_bytes = _make_pdf_bytes("AI Analyst — Recommendations", auto_brief)
st.download_button(
    "📄 Download AI summary (PDF)",
    data=pdf_bytes,
    file_name="ai_analyst_summary.pdf",
    mime="application/pdf",
    use_container_width=True
)

st.subheader("Ask your own question")
if "sa_chat" not in st.session_state:
    st.session_state.sa_chat = []
q = st.text_input("Ask:")
if st.button("Ask AI") and q.strip():
    st.session_state.sa_chat.append(("you", q))
    q_text = json.dumps(ai_summary, default=sa_numpy_converter)
    reply = sa_call_llm(ai_system, f"Context:\n{q_text}\nQuestion: {q}", ai_cfg)
    st.session_state.sa_chat.append(("ai", reply))
for role, msg in st.session_state.sa_chat:
    st.markdown(f"**{role.title()}:** {msg}")

st.caption("Privacy: Only summary JSON is sent to AI; raw data is never shared.")
