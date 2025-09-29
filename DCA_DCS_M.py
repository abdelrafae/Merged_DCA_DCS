
import io, base64, textwrap, pathlib, math
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid as cumtrapz
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

from scipy.integrate import cumulative_trapezoid as cumtrapz


# ---------------------------------------------------------------------
# App & Theme
# ---------------------------------------------------------------------
st.set_page_config(page_title="Hyperbolic Decline + Monte Carlo", layout="wide", page_icon="📉")

cfg_dir = pathlib.Path(".streamlit"); cfg_dir.mkdir(exist_ok=True)
(cfg_dir / "config.toml").write_text(textwrap.dedent("""\
[theme]
base = "light"
primaryColor = "#2748d9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F6F8FF"
textColor = "#1f2a44"
"""), encoding="utf-8")

with st.sidebar:
    st.markdown("### 🎨 Appearance")
    accent_choice = st.selectbox("Accent color", ["Blue","Teal","Purple","Emerald"], index=0)

ACCENTS = {
   "Blue":    {"accent":"#2748d9", "accentSoft":"#eef3ff"},
   "Teal":    {"accent":"#0e8f8c", "accentSoft":"#e9fbfb"},
   "Purple":  {"accent":"#6d28d9", "accentSoft":"#f3e9ff"},
   "Emerald": {"accent":"#059669", "accentSoft":"#e9fbf4"},
}
ACCENT = ACCENTS[accent_choice]["accent"]
ACCENT_SOFT = ACCENTS[accent_choice]["accentSoft"]

# nice KPI cards for the 4 tiles
st.markdown("""
<style>
.k4 {display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:16px;margin:8px 0 10px;}
.kbox{background:#fff;border:1px solid #e6e9f3;border-radius:14px;padding:14px 16px;box-shadow:0 4px 12px rgba(36,71,187,.06);}
.kl{font-size:11px;color:#6b7a99;text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px 0;}
.kv{font-size:24px;font-weight:800;line-height:1.15;}
.ku{font-size:11px;color:#6b7a99;margin-left:6px;font-weight:700;}
.ks{font-size:11px;margin-top:2px;color:#6b7a99;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: Data
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Hyperbolic Decline")
    uploaded_file = st.file_uploader("📂 Upload CSV (columns: wellname, date, oil, days)", type=["csv"])
    st.markdown("---")
    st.markdown("#### ⚙️ Column Mapping & Bounds")

if uploaded_file:
    @st.cache_data(show_spinner=False)
    def load_data(file): return pd.read_csv(file)

    df_raw = load_data(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")

    with st.sidebar:
        well_col = st.selectbox("🛢️ Well Column", df_raw.columns, key="well_col")
        date_col = st.selectbox("📅 Date Column", df_raw.columns, key="date_col")
        oil_col  = st.selectbox("🛢️ Total Oil Column", df_raw.columns, key="oil_col")
        days_col = st.selectbox("📆 Days Column", df_raw.columns, key="days_col")
        oil_rate_col = st.text_input("🆕 New Oil-Rate Column Name", "oil_rate", key="oil_rate_name")
        b_min = st.number_input("🔽 Min b", value=0.00, step=0.01, key="b_min")
        b_max = st.number_input("🔼 Max b", value=1.00, step=0.01, key="b_max")
        submitted = st.button("✅ Apply", use_container_width=True, key="apply_btn")

    if submitted:
        df = df_raw[[well_col, date_col, oil_col, days_col]].copy()
        df.columns = ["well", "date", "oil", "days"]
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            st.error(f"❌ Date conversion error: {e}"); st.stop()
        df["oil"] = pd.to_numeric(df["oil"], errors="coerce")
        df["days"]= pd.to_numeric(df["days"],errors="coerce")
        df[oil_rate_col] = df["oil"] / df["days"]
        df = df[(df[oil_rate_col] > 0) & df[oil_rate_col].notna()]
        df.dropna(subset=["well","date",oil_rate_col], inplace=True)
        df.sort_values(by=["well","date"], inplace=True)
        st.session_state.data_ready = df
        st.session_state.b_range = (b_min, b_max)
        st.session_state.oil_rate_col = oil_rate_col

st.markdown("<div class='card'><h3>Main Dashboard</h3></div>", unsafe_allow_html=True)
if "data_ready" not in st.session_state:
    st.info("Upload a CSV and click Apply.")
    st.stop()

df = st.session_state.data_ready
b_min, b_max = st.session_state.b_range
oil_rate_col = st.session_state.oil_rate_col
wells = sorted(df["well"].unique())

tab_overview, tab_single, tab_batch = st.tabs(["📈 Overview", "🛢️ Single Well Fit", "🧮 All Wells Summary"])

# ---------------------------------------------------------------------
# Decline helpers
# ---------------------------------------------------------------------
def arps_rate(qi, di, b, t):
    if b <= 1e-10:
        return qi * np.exp(-di * t)
    return qi / (1 + b * di * t) ** (1.0 / b)

def _weighted_mse_log(q_pred, q_data):
    w = np.linspace(1, 3, len(q_data))
    return np.mean(w * (np.log1p(q_pred) - np.log1p(q_data))**2)

def _loss_core(q_pred, q_data, qe_target, b_list, seg_count, complexity_lambda=0.22):
    mse_log = _weighted_mse_log(q_pred, q_data)
    qe_pen  = (abs(q_pred[-1] - qe_target) / max(qe_target, 1))**2 * 35.0
    b_pen   = sum((max(b - 1, 0))**2 * 1.6 for b in b_list)
    comp    = complexity_lambda * (seg_count - 1)
    return mse_log + qe_pen + b_pen + comp

def _break_candidates_from_curvature(t, q, n_desired=24, min_pts=3):
    n = len(q)
    if n < 2*min_pts+1: return np.array([], dtype=int)
    logq = np.log(np.clip(q, 1e-8, None))
    k = max(3, min(9, n//12))
    smooth = np.convolve(logq, np.ones(k)/k, mode="same")
    d2 = np.zeros_like(smooth); d2[1:-1] = smooth[2:] - 2*smooth[1:-1] + smooth[:-2]
    order = np.argsort(-np.abs(d2))
    interior = [i for i in order if (i>=min_pts and i<=n-min_pts-1)]
    picks, guard = [], max(1, n//30)
    for idx in interior:
        if all(abs(idx - p) >= guard for p in picks):
            picks.append(idx)
        if len(picks) >= n_desired: break
    picks = np.array(sorted(set(picks)))
    if len(picks) < max(2, n_desired//2):
        extra = np.linspace(min_pts, n-min_pts-1, num=max(2, n_desired//2), dtype=int)
        picks = np.unique(np.concatenate([picks, extra]))
    return picks

def _exp_init(t, q):
    q = np.clip(q, 1e-6, None); t = t - t[0]
    if len(t) < 2: return 0.02, 0.3
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, np.log(q), rcond=None)[0]
    di0 = max(1e-4, min(0.3, -slope)); b0 = 0.3
    return di0, b0

def _fit_segment(qi0, t_seg, q_seg, b_bounds, init=None):
    bmin, bmax = b_bounds
    di0, b0 = _exp_init(t_seg, q_seg) if init is None else init
    bounds = [(1e-5, 0.6), (bmin, bmax)]
    def seg_loss(x):
        di, b = x
        if di <= 0 or b < bmin or b > bmax: return 1e9
        q_pred = arps_rate(qi0, di, b, t_seg - t_seg[0])
        if np.any(q_pred <= 0) or np.any(np.isnan(q_pred)): return 1e9
        return _weighted_mse_log(q_pred, q_seg)
    res = minimize(seg_loss, x0=np.array([di0, b0]), method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=200, ftol=1e-9))
    di, b = res.x
    q_pred = arps_rate(qi0, di, b, t_seg - t_seg[0])
    return float(di), float(b), q_pred

def _assemble_model(Qi, t, q, b_bounds, break_idxs, complexity_lambda=0.22):
    indices = [0] + [int(i)+1 for i in sorted(break_idxs)] + [len(t)]
    segments, preds = [], []
    qi_cur = Qi
    for s in range(len(indices)-1):
        i0, i1 = indices[s], indices[s+1]
        t_seg = t[i0:i1]; q_seg = q[i0:i1]
        di, b, q_pred = _fit_segment(qi_cur, t_seg, q_seg, b_bounds)
        segments.append({"di": float(di), "b": float(b)})
        preds.append(q_pred)
        qi_cur = q_pred[-1]
    q_fit = np.concatenate(preds)
    loss = _loss_core(q_fit, q, q[-1], [s["b"] for s in segments], len(segments),
                      complexity_lambda=complexity_lambda)
    return {"segments": segments, "break_idxs": sorted(break_idxs), "q_fit": q_fit, "loss": loss}

def fit_auto_greedy(Qi, t, q, b_bounds,
                    max_segments=5, min_improvement=0.01, min_segments=2,
                    complexity_lambda=0.22, n_candidates=24, min_pts=3):
    n = len(q)
    if n < 4:
        return _assemble_model(Qi, t, q, b_bounds, [], complexity_lambda)
    candidates = list(_break_candidates_from_curvature(t, q, n_desired=n_candidates, min_pts=min_pts))
    model = _assemble_model(Qi, t, q, b_bounds, [], complexity_lambda)
    if max_segments <= 1 or len(candidates) == 0:
        return model
    def valid_with(current, new_idx):
        all_idx = sorted(current + [new_idx])
        edges = [-1] + all_idx + [n-1]
        return all((b - a) >= (min_pts+1) for a,b in zip(edges, edges[1:]))
    while len(model["segments"]) < max_segments:
        best_try = None
        for tb in candidates:
            if tb in model["break_idxs"]: continue
            if not valid_with(model["break_idxs"], tb): continue
            try_model = _assemble_model(Qi, t, q, b_bounds, model["break_idxs"] + [tb], complexity_lambda)
            if (best_try is None) or (try_model["loss"] < best_try["loss"]):
                best_try = try_model
        if best_try is None: break
        rel_impr = (model["loss"] - best_try["loss"]) / max(model["loss"], 1e-12)
        if len(model["segments"]) < (min_segments):
            model = best_try; continue
        if rel_impr >= min_improvement: model = best_try
        else: break
    return model

# MC helpers
def time_to_qe(qi, Di, b, qe):
    if Di <= 0 or qi <= 0 or qe <= 0 or qe >= qi: return np.nan
    if abs(b) < 1e-8:   return (1.0/Di)*math.log(qi/qe)
    if abs(b-1) < 1e-8: return (qi/qe - 1.0)/Di
    return ((qi/qe)**b - 1.0) / (b*Di)

def _cum_to_time(qi, Di, b, t):
    if t <= 0: return 0.0
    if abs(b) < 1e-8:      return qi * (1 - np.exp(-Di * t)) / Di
    if abs(b - 1.0) < 1e-8:return (qi / Di) * np.log(1 + Di * t)
    return (qi / ((1.0 - b) * Di)) * ((1.0 + b * Di * t) ** ((1.0 - b) / b) - 1.0)

def _piecewise_time_cum_to_qe(qi0_m, qe_m, seg_params, seg_durations):
    t_total, np_total = 0.0, 0.0
    qi_cur = qi0_m
    for (Di,b), Tseg in zip(seg_params, seg_durations):
        if qi_cur <= qe_m: return t_total, np_total, True
        t_hit = time_to_qe(qi_cur, Di, b, qe_m)
        if np.isfinite(t_hit) and t_hit < Tseg:
            np_total += _cum_to_time(qi_cur, Di, b, t_hit)
            t_total += t_hit
            return t_total, np_total, True
        np_total += _cum_to_time(qi_cur, Di, b, Tseg)
        t_total += Tseg
        qi_cur = arps_rate(qi_cur, Di, b, Tseg)
    return t_total, np_total, False

def _build_profile_until_qe(qi0_m, qe_m, seg_params, seg_durations, dt=0.25):
    ts=[0.0]; qs=[qi0_m]; t_acc=0.0; qi_cur=qi0_m
    for (Di,b),Tseg in zip(seg_params, seg_durations):
        steps=max(1,int(Tseg/dt))
        for k in range(1, steps+1):
            t_local=min(k*dt, Tseg)
            q_now=arps_rate(qi_cur, Di, b, t_local)
            ts.append(t_acc+t_local); qs.append(q_now)
            if q_now<=qe_m: return np.array(ts), np.array(qs)
        t_acc+=Tseg
        qi_cur=arps_rate(qi_cur, Di, b, Tseg)
    return np.array(ts), np.array(qs)

def p10_p50_p90_volume(arr):
    return (np.nanpercentile(arr,90), np.nanpercentile(arr,50), np.nanpercentile(arr,10))

# distributions (Fixed or Min/Max)
def sample_from_spec(spec: dict, n: int) -> np.ndarray:
    kind = spec["kind"]
    if kind == "Fixed":
        v = float(spec["value"])
        return np.full(n, v, dtype=float)
    vmin = float(spec["min"]); vmax = float(spec["max"])
    if vmin > vmax: raise ValueError("Min must be ≤ Max.")
    if abs(vmax - vmin) < 1e-12: return np.full(n, vmin, dtype=float)
    if kind == "Uniform":
        return np.random.uniform(vmin, vmax, size=n)
    elif kind == "Triangular":
        mode = 0.5 * (vmin + vmax); return np.random.triangular(vmin, mode, vmax, size=n)
    elif kind == "PERT (Beta-PERT)":
        x = np.random.beta(3.0, 3.0, size=n); return vmin + x * (vmax - vmin)
    elif kind == "Normal (Truncated)":
        mu = 0.5 * (vmin + vmax); sigma = (vmax - vmin) / 4.0
        x = np.random.normal(mu, sigma, size=n); return np.clip(x, vmin, vmax)
    elif kind == "Lognormal (from bounds)":
        _min = max(1e-12, vmin); _max = max(_min + 1e-12, vmax)
        mu = math.log((_min * _max) ** 0.5); sigma = (math.log(_max) - math.log(_min)) / 4.0
        x = np.random.lognormal(mean=mu, sigma=max(1e-6, sigma), size=n); return np.clip(x, vmin, vmax)
    elif kind == "Beta (scaled)":
        x = np.random.beta(2.0, 2.0, size=n); return vmin + x * (vmax - vmin)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")

def dist_block(label, default_kind="Fixed", default_min=0.0, default_max=1.0, default_value=1.0, key_prefix=""):
    kinds = ["Fixed","Uniform","Triangular","PERT (Beta-PERT)","Normal (Truncated)","Lognormal (from bounds)","Beta (scaled)"]
    st.markdown(f"##### {label}")
    kind = st.selectbox(f"{label} distribution", kinds, index=kinds.index(default_kind), key=f"{key_prefix}_kind")
    if kind == "Fixed":
        value = st.text_input(f"{label} Value", value=str(default_value), key=f"{key_prefix}_value")
        return {"kind": kind, "value": value}
    else:
        c1,c2 = st.columns(2)
        vmin = c1.text_input(f"{label} Min", value=str(default_min), key=f"{key_prefix}_min")
        vmax = c2.text_input(f"{label} Max", value=str(default_max), key=f"{key_prefix}_max")
        return {"kind": kind, "min": vmin, "max": vmax}

# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
with tab_overview:
    c1,c2,c3 = st.columns([1,1,1])
    c1.metric("Total Wells", len(wells))
    c2.metric("Records Loaded", f"{len(df):,}")
    span_days = int((df['date'].max() - df['date'].min()).days)
    c3.metric("Time Span (days)", f"{span_days:,}")

# ---------------------------------------------------------------------
# SINGLE WELL
# ---------------------------------------------------------------------
with tab_single:
    st.markdown("<div class='card'><h4>Controls</h4>", unsafe_allow_html=True)
    cA,cB,cC,cD = st.columns([1.2,1.0,1.0,1.0])
    with cA: selected_well = st.selectbox("Select Well", wells, key="selected_well")
    with cB: mode = st.selectbox("Model Mode", ["Automatic (≥2 segments)", "Single only"], index=1)
    with cC: max_segments = st.number_input("Max segments (2–10)", 2, 10, 5, 1)
    with cD: min_segments = st.number_input("Min segments", 1, 10, 2, 1)
    col1,col2,col3 = st.columns([1.15,1.0,1.0])
    with col1: min_improvement = st.slider("Min improvement to add a segment", 0.0, 0.10, 0.01, 0.01)
    with col2: complexity_lambda = st.slider("Aggressiveness (penalty)", 0.00, 1.00, 0.22, 0.02)
    with col3: n_candidates = st.slider("Breakpoint candidates", 6, 36, 24, 2)
    st.caption("Default mode is **Single**. Switch to Auto if you want ≥2 segments.")
    st.markdown("</div>", unsafe_allow_html=True)

    # data for this well
    wd = df[df["well"] == selected_well].copy()
    qi_idx = wd[oil_rate_col].idxmax()
    qi_date = wd.loc[qi_idx, "date"]
    Qi = wd.loc[qi_idx, oil_rate_col]
    before_qi = wd[wd["date"] < qi_date]
    after_qi = wd[wd["date"] >= qi_date].copy()
    after_qi["t_months"] = (after_qi["date"] - after_qi["date"].iloc[0]).dt.days / 30.4375
    t_data = after_qi["t_months"].values
    q_data = after_qi[oil_rate_col].values
    Qe = q_data[-1]

    # fit
    if mode.startswith("Single"):
        model = _assemble_model(Qi, t_data, q_data, (b_min, b_max), [], complexity_lambda)
    else:
        model = fit_auto_greedy(Qi, t_data, q_data, (b_min, b_max),
                                max_segments=int(max_segments),
                                min_improvement=float(min_improvement),
                                min_segments=int(min_segments),
                                complexity_lambda=float(complexity_lambda),
                                n_candidates=int(n_candidates))

    q_fit = model["q_fit"]
    after_qi["fitted_rate"] = q_fit
    after_qi["cum_fitted"] = after_qi["fitted_rate"].cumsum()

    # plot fit
    chart_col, table_col = st.columns([1.4, 1.0], gap="large")
    with chart_col:
        st.markdown("<div class='card canvas-tall'><h4>Decline Curve</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11.4,5.8), dpi=110)
        ax.scatter(before_qi["date"], before_qi[oil_rate_col], s=22, label="Pre-Qi")
        ax.scatter(after_qi["date"], after_qi[oil_rate_col], s=22, label="Post-Qi")
        ax.plot(after_qi["date"], q_fit, lw=2.2, label="Fitted")
        ax.axvline(qi_date, ls="--", c="k", alpha=.5); ax.text(qi_date, ax.get_ylim()[1]*.96, "Qi", rotation=90, va="top")
        for tb in model["break_idxs"]:
            d = after_qi["date"].iloc[tb]
            ax.axvline(d, ls="--", c="gray", alpha=.6)
        ax.set_xlabel("Date"); ax.set_ylabel("Rate (STB/day)"); ax.grid(linestyle="--", alpha=.25); ax.legend(frameon=False)
        st.pyplot(fig); st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown("<div class='card'><h4>Decline Table</h4>", unsafe_allow_html=True)
        show = after_qi[["date", oil_rate_col, "fitted_rate", "cum_fitted"]].rename(columns={oil_rate_col:"oil_rate"}).copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(show, use_container_width=True, height=480)
        buf = BytesIO(); after_qi.to_excel(buf, index=False)
        st.download_button("📥 Download Fit (Excel)", buf.getvalue(), file_name=f"{selected_well}_fit.xlsx", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Monte Carlo for this well ----------------
    st.markdown("<div class='card'><h4>Monte Carlo — This Well</h4>", unsafe_allow_html=True)
    segs = model["segments"]
    if len(segs) == 0:
        st.info("No segments to simulate.")
    else:
        # segment durations (months) — extend last
        idxs = [0] + [int(i)+1 for i in sorted(model["break_idxs"])] + [len(t_data)]
        seg_durations = [ float(t_data[idxs[s+1]-1] - t_data[idxs[s]]) for s in range(len(idxs)-1) ]
        seg_durations[-1] = 10000.0

        Qi_base_d = float(Qi)
        Qe_base_d = float(q_data[-1])
        days_per_month = 30.4375

        c1,c2 = st.columns(2)
        with c1: n_trials = st.number_input("Simulations", 1000, 2_000_000, 50000, step=5000)
        with c2: seed_one = st.number_input("Random seed (−1=random)", -1, 9999999, 42)
        if seed_one >= 0: np.random.seed(int(seed_one))

        # distributions with sensible defaults from fit
        st.markdown("**Inputs (Fixed or Min/Max)**")
        ci, cq = st.columns(2)
        with ci:
            qi_spec = dist_block("Qi (STB/day)", default_kind="Fixed",
                                 default_value=Qi_base_d, default_min=max(1.0, Qi_base_d*0.5), default_max=Qi_base_d*1.5,
                                 key_prefix="one_qi")
            di_spec = dist_block("Di (per month) — applies to all segments", default_kind="Fixed",
                                 default_value=np.mean([s['di'] for s in segs]), default_min=0.005, default_max=0.6,
                                 key_prefix="one_di")
        with cq:
            qe_spec = dist_block("Qe (STB/day)", default_kind="Fixed",
                                 default_value=Qe_base_d, default_min=max(0.1, Qe_base_d*0.5), default_max=Qe_base_d*1.5,
                                 key_prefix="one_qe")
            b_spec  = dist_block("b (applies to all segments)", default_kind="Fixed",
                                 default_value=np.mean([s['b'] for s in segs]), default_min=0.2, default_max=1.5,
                                 key_prefix="one_b")

        # reference table
        tbl = pd.DataFrame([{"Seg":i+1, "Di_base_/month": s["di"], "b_base": s["b"], "Duration_months": seg_durations[i]}
                            for i,s in enumerate(segs)])
        st.dataframe(tbl, use_container_width=True)

        run_one = st.button("🎲 Run Monte Carlo (this well)", use_container_width=True)
        if run_one:
            n = int(n_trials)

            # draws
            qi_draw_d = sample_from_spec(qi_spec, n)
            qe_draw_d = sample_from_spec(qe_spec, n)
            qi_draw_m = qi_draw_d * days_per_month
            qe_draw_m = qe_draw_d * days_per_month
            di_draw = sample_from_spec(di_spec, n)
            b_draw  = sample_from_spec(b_spec,  n)
            b_draw  = np.maximum(b_draw, 0.2)
            nseg = len(segs)
            Di_draws = np.repeat(di_draw.reshape(-1,1), nseg, axis=1)
            b_draws  = np.repeat(b_draw.reshape(-1,1),  nseg, axis=1)
            qe_draw_m = np.minimum(qe_draw_m, 0.95*qi_draw_m)

            # run MC
            Tm = np.empty(n); Np = np.empty(n)
            for i in range(n):
                params_i = list(zip(Di_draws[i], b_draws[i]))
                Tm[i], Np[i], _ = _piecewise_time_cum_to_qe(qi_draw_m[i], qe_draw_m[i], params_i, seg_durations)


            # --- Keep only valid trials once ---
            valid = np.isfinite(Tm) & np.isfinite(Np) & (Tm >= 0) & (Np >= 0)
            if not np.any(valid):
                st.error("All trials invalid. Check inputs.")
                st.stop()

            Np_v = Np[valid]
            Tm_v = Tm[valid]

            # helper: index of trial whose NP is closest to the percentile target
            def idx_for_np_percentile(np_all_valid, full_mask, p):
                target = np.nanpercentile(np_all_valid, p)
                # map back to full-length array index
                # find in the valid subset first
                i_valid = int(np.nanargmin(np.abs(np_all_valid - target)))
                # convert 'valid index' to 'full array index'
                full_indices = np.flatnonzero(full_mask)
                return int(full_indices[i_valid])

            i_p10 = idx_for_np_percentile(Np_v, valid, 90)  # Optimistic = high NP
            i_p50 = idx_for_np_percentile(Np_v, valid, 50)  # Median
            i_p90 = idx_for_np_percentile(Np_v, valid, 10)  # Conservative = low NP

            # Representative NP and times (T @ NP) taken directly from those same trials
            np_p10, t_at_p10 = float(Np[i_p10]), float(Tm[i_p10])
            np_p50, t_at_p50 = float(Np[i_p50]), float(Tm[i_p50])
            np_p90, t_at_p90 = float(Np[i_p90]), float(Tm[i_p90])

            # ---- Base case (fit) ----
            base_params = [(s["di"], s["b"]) for s in segs]
            t_base, np_base, _ = _piecewise_time_cum_to_qe(
                Qi_base_d * days_per_month,
                Qe_base_d * days_per_month,
                base_params,
                seg_durations
            )

            # ---- KPI cards (NP and T @ NP) ----
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='kpi-box'><div class='kpi-label'>NP P10 (optimistic)</div><div class='kpi-value'>{np_p10:,.0f}<span class='kpi-unit'> STB</span></div><div class='kpi-sub'>T @ NP: {t_at_p10:,.2f} months</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi-box'><div class='kpi-label'>NP P50 (median)</div><div class='kpi-value'>{np_p50:,.0f}<span class='kpi-unit'> STB</span></div><div class='kpi-sub'>T @ NP: {t_at_p50:,.2f} months</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi-box'><div class='kpi-label'>NP P90 (conservative)</div><div class='kpi-value'>{np_p90:,.0f}<span class='kpi-unit'> STB</span></div><div class='kpi-sub'>T @ NP: {t_at_p90:,.2f} months</div></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='kpi-box'><div class='kpi-label'>NP Base</div><div class='kpi-value'>{np_base:,.0f}<span class='kpi-unit'> STB</span></div><div class='kpi-sub'>T @ NP: {t_base:,.2f} months</div></div>", unsafe_allow_html=True)

            # ---- Profiles for those same trials (q vs t) and cumulative NP(t) ----
            def profile_until_qe_from_index(i):
                params_i = list(zip(Di_draws[i], b_draws[i]))
                return _build_profile_until_qe(qi_draw_m[i], qe_draw_m[i], params_i, seg_durations, dt=0.25)

            # Build the 3 reps in the order we want the colors to be: blue, orange, green
            rep_defs = [
                ("Conservative (P90)", i_p10),     # blue
                ("Median (P50)",     i_p50),     # orange
                ("Optimistic (P10)", i_p90),   # green
            ]

            import matplotlib.pyplot as plt
            from scipy.integrate import cumulative_trapezoid as cumtrapz

            fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.4), dpi=110)

            # Left: q vs t
            for label, idx in rep_defs:
                tt, qq = profile_until_qe_from_index(idx)
                axL.plot(tt, qq, label=label)
            # Base fit line (dashed)
            tt_b, qq_b = _build_profile_until_qe(Qi_base_d*days_per_month, Qe_base_d*days_per_month,
                                                 base_params, seg_durations, dt=0.25)
            axL.plot(tt_b, qq_b, "k--", alpha=0.8, label="Base (fit)")
            axL.axhline(Qe_base_d*days_per_month, color="gray", ls="--", alpha=0.45)
            axL.set_xlabel("Time (months)")
            axL.set_ylabel("q (STB/month)")
            axL.legend(frameon=False)

            # Right: cumulative NP(t)
            for label, idx in rep_defs:
                tt, qq = profile_until_qe_from_index(idx)
                cum = cumtrapz(qq, tt, initial=0.0)  # STB
                axR.plot(tt, cum, label=label)
            cum_base = cumtrapz(qq_b, tt_b, initial=0.0)
            axR.plot(tt_b, cum_base, "k--", alpha=0.8, label="Base (fit)")
            axR.set_xlabel("Time (months)")
            axR.set_ylabel("Cumulative NP (STB)")
            axR.legend(frameon=False)

            fig.suptitle(f"{selected_well} — Decline (left) and Cumulative (right)")
            st.pyplot(fig)

            # ---- Excel export for this well ----
            out_df = pd.DataFrame([{
                "well": selected_well, "Segments": len(segs),
                "Np_P10_STB": np_p10, "T_at_NP_P10_months": t_at_p10,
                "Np_P50_STB": np_p50, "T_at_NP_P50_months": t_at_p50,
                "Np_P90_STB": np_p90, "T_at_NP_P90_months": t_at_p90,
                "Np_Base_STB": np_base, "T_at_NP_Base_months": t_base,
            }]).round(4)

            out_buf = BytesIO()
            with pd.ExcelWriter(out_buf, engine="xlsxwriter") as wr:
                out_df.to_excel(wr, sheet_name=f"{selected_well}_MC", index=False)

            st.download_button("📥 Download MC (this well)", out_buf.getvalue(),
                               file_name=f"{selected_well}_monte_carlo_p10_p50_p90.xlsx",
                               use_container_width=True)


# ---------------------------------------------------------------------
# ALL-WELLS SUMMARY + MC EXPORT
# ---------------------------------------------------------------------
with tab_batch:
    st.markdown("<div class='card'><h4>All-Wells Fitting Summary</h4>", unsafe_allow_html=True)
    if st.button("⚙️ Compute All-Wells Fitting Table", use_container_width=True):
        results=[]; wells_all=sorted(df["well"].unique()); progress=st.progress(0); status=st.empty()
        for i,w in enumerate(wells_all, start=1):
            status.info(f"Fitting **{w}** ({i}/{len(wells_all)}) …")
            try:
                wdf=df[df["well"]==w].copy()
                qi_idx_w=wdf[oil_rate_col].idxmax(); qi_date_w=wdf.loc[qi_idx_w,"date"]; Qi_w=wdf.loc[qi_idx_w,oil_rate_col]
                aft=wdf[wdf["date"]>=qi_date_w].copy()
                aft["t_months"]=(aft["date"]-aft["date"].iloc[0]).dt.days/30.4375
                t_w=aft["t_months"].values; q_w=aft[oil_rate_col].values
                if len(t_w)<4: results.append({"well":w,"segments":0,"status":"insufficient"}); progress.progress(i/len(wells_all)); continue
                # default Single
                model_w=_assemble_model(Qi_w, t_w, q_w, (b_min,b_max), [], 0.22)
                segs_w=model_w["segments"]; indices=[0]+[int(ix)+1 for ix in sorted(model_w["break_idxs"])]+[len(t_w)]
                seg_starts=[]; seg_ends=[]
                for j in range(len(indices)-1):
                    s=indices[j]; e=indices[j+1]-1
                    seg_starts.append(aft["date"].iloc[s].date()); seg_ends.append(aft["date"].iloc[e].date())
                row={"well":w,"segments":len(segs_w)}
                for j in range(10):
                    row[f"Di_{j+1}"]=segs_w[j]["di"] if j<len(segs_w) else np.nan
                    row[f"b_{j+1}"]=segs_w[j]["b"] if j<len(segs_w) else np.nan
                    row[f"Seg{j+1}_Start"]=seg_starts[j] if j<len(seg_starts) else None
                    row[f"Seg{j+1}_End"]=seg_ends[j] if j<len(seg_ends) else None
                results.append(row)
            except Exception as e:
                results.append({"well":w,"segments":0,"status":f"error: {e}"})
            progress.progress(i/len(wells_all))
        status.success("✅ Done.")
        st.session_state.fit_summary = pd.DataFrame(results)
    if "fit_summary" in st.session_state:
        st.dataframe(st.session_state.fit_summary, use_container_width=True, height=520)
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
            st.session_state.fit_summary.to_excel(wr, sheet_name="summary", index=False)
        st.download_button("📥 Download Summary (Excel)", buf.getvalue(), file_name="all_wells_fitting_summary.xlsx", use_container_width=True)

    # MC export for ALL wells (default Single)
    st.markdown("<div class='card'><h4>Monte Carlo — Export P10/P50/P90 for ALL Wells</h4>", unsafe_allow_html=True)
    seg_mode_all = st.selectbox("Segment mode for Monte Carlo (export)",
                                ["Automatic (≥2 segments)", "Single segment"], index=1)

    def ui_spec_global(title, kind_default="Fixed", value_default="", min_default="", max_default="", key=""):
        kinds = ["Fixed","Uniform","Triangular","PERT (Beta-PERT)","Normal (Truncated)","Lognormal (from bounds)","Beta (scaled)"]
        kind = st.selectbox(f"{title} distribution", kinds, index=kinds.index(kind_default), key=f"all_{key}_kind")
        if kind == "Fixed":
            val = st.text_input(f"{title} Value", value=str(value_default), key=f"all_{key}_value")
            return {"kind":"Fixed","value":val}
        c1,c2 = st.columns(2)
        vmin = c1.text_input(f"{title} Min", value=str(min_default), key=f"all_{key}_min")
        vmax = c2.text_input(f"{title} Max", value=str(max_default), key=f"all_{key}_max")
        return {"kind":kind,"min":vmin,"max":vmax}

    c0,c1 = st.columns([1,1])
    with c0: sims_all = st.number_input("Simulations per well", 1000, 2_000_000, 30000, step=5000)
    with c1: seed_all = st.number_input("Random seed (−1=random)", -1, 9999999, 7)
    if seed_all >= 0: np.random.seed(int(seed_all))

    g1,g2 = st.columns(2)
    with g1:
        qi_spec_all = ui_spec_global("Qi (STB/day)", kind_default="Fixed", key="qi")
        di_spec_all = ui_spec_global("Di (/month)",   kind_default="Fixed", key="di")
    with g2:
        qe_spec_all = ui_spec_global("Qe (STB/day)", kind_default="Fixed", key="qe")
        b_spec_all  = ui_spec_global("b",            kind_default="Fixed", key="b")

    # run_all = st.button("📥 Run & Export Monte Carlo for ALL Wells", use_container_width=True)


    run_all = st.button("📥 Run & Export Monte Carlo for ALL Wells", use_container_width=True)
    if run_all:
        rows = []
        wells_all = sorted(df["well"].unique())
        prog = st.progress(0)
        stat = st.empty()

        def idx_for_np_percentile(np_all_valid, full_mask, p):
            """Return index in the ORIGINAL arrays that corresponds to the trial whose
            NP is closest to the percentile p of the *valid* NP distribution."""
            target = np.nanpercentile(np_all_valid, p)
            i_valid = int(np.nanargmin(np.abs(np_all_valid - target)))
            full_indices = np.flatnonzero(full_mask)
            return int(full_indices[i_valid])

        dpm = 30.4375

        for i, w in enumerate(wells_all, start=1):
            stat.info(f"MC for **{w}** ({i}/{len(wells_all)}) …")
            try:
                wdf = df[df["well"] == w].copy()

                # base (fit) inputs per well
                qi_idx_w  = wdf[oil_rate_col].idxmax()
                qi_date_w = wdf.loc[qi_idx_w, "date"]
                Qi_base_d = float(wdf.loc[qi_idx_w, oil_rate_col])

                aft = wdf[wdf["date"] >= qi_date_w].copy()
                aft["t_months"] = (aft["date"] - aft["date"].iloc[0]).dt.days / dpm
                t_w = aft["t_months"].values
                q_w = aft[oil_rate_col].values
                if len(t_w) < 4:
                    rows.append({"well": w, "error": "insufficient data"})
                    prog.progress(i / len(wells_all))
                    continue

                # Segment mode (default SINGLE)
                if seg_mode_all.startswith("Single"):
                    model_w = _assemble_model(Qi_base_d, t_w, q_w, (b_min, b_max), [], 0.22)
                else:
                    model_w = fit_auto_greedy(Qi_base_d, t_w, q_w, (b_min, b_max),
                                              max_segments=5, min_improvement=0.01, min_segments=2,
                                              complexity_lambda=0.22, n_candidates=24)

                segs_w = model_w["segments"]
                if len(segs_w) == 0:
                    rows.append({"well": w, "error": "fit failed"})
                    prog.progress(i / len(wells_all))
                    continue

                # segment durations; extend last to allow reaching Qe
                idxs = [0] + [int(ix) + 1 for ix in sorted(model_w["break_idxs"])] + [len(t_w)]
                seg_durations = [float(t_w[idxs[s + 1] - 1] - t_w[idxs[s]]) for s in range(len(idxs) - 1)]
                seg_durations[-1] = 10000.0

                # per-well defaults for Fixed
                defaults = {
                    "qi": Qi_base_d,
                    "qe": float(q_w[-1]),
                    "di": float(np.mean([s["di"] for s in segs_w])),
                    "b" : float(np.mean([s["b"]  for s in segs_w])),
                }
                def fill_spec(spec, key):
                    if spec["kind"] == "Fixed" and (spec["value"] == "" or spec["value"] is None):
                        return {"kind": "Fixed", "value": defaults[key]}
                    return spec

                qi_spec_w = fill_spec(qi_spec_all, "qi")
                qe_spec_w = fill_spec(qe_spec_all, "qe")
                di_spec_w = fill_spec(di_spec_all, "di")
                b_spec_w  = fill_spec(b_spec_all,  "b")

                # ------- Monte Carlo draws -------
                n = int(sims_all)
                qi_draw_d = sample_from_spec(qi_spec_w, n)
                qe_draw_d = sample_from_spec(qe_spec_w, n)
                qi_m = qi_draw_d * dpm
                qe_m = qe_draw_d * dpm

                di_draw = sample_from_spec(di_spec_w, n)
                b_draw  = sample_from_spec(b_spec_w,  n)
                b_draw  = np.maximum(b_draw, 0.2)

                nseg = len(segs_w)
                Di_draws = np.repeat(di_draw.reshape(-1, 1), nseg, axis=1)
                b_draws  = np.repeat(b_draw.reshape(-1, 1),  nseg, axis=1)

                # ensure qe < qi
                qe_m = np.minimum(qe_m, 0.95 * qi_m)

                # run trials
                Tm = np.empty(n)
                Np = np.empty(n)
                for k in range(n):
                    params = list(zip(Di_draws[k], b_draws[k]))
                    Tm[k], Np[k], _ = _piecewise_time_cum_to_qe(qi_m[k], qe_m[k], params, seg_durations)

                valid = (np.isfinite(Tm) & np.isfinite(Np) & (Tm >= 0) & (Np >= 0))
                if not np.any(valid):
                    rows.append({"well": w, "error": "no valid trials"})
                    prog.progress(i / len(wells_all))
                    continue

                # NP percentiles + matching times (same trial!)
                Np_v = Np[valid]
                i10 = idx_for_np_percentile(Np_v, valid, 90)  # optimistic
                i50 = idx_for_np_percentile(Np_v, valid, 50)
                i90 = idx_for_np_percentile(Np_v, valid, 10)  # conservative
                np_p10, t_at_p10 = float(Np[i10]), float(Tm[i10])
                np_p50, t_at_p50 = float(Np[i50]), float(Tm[i50])
                np_p90, t_at_p90 = float(Np[i90]), float(Tm[i90])

                # -------- Base (fit) NP and T --------
                base_params = [(s["di"], s["b"]) for s in segs_w]
                t_base_w, np_base_w, _ = _piecewise_time_cum_to_qe(
                    Qi_base_d * dpm,
                    defaults["qe"] * dpm,
                    base_params,
                    seg_durations
                )

                rows.append({
                    "well": w,
                    "Segments": nseg,
                    "Np_P10_STB": np_p10, "T_at_NP_P10_months": t_at_p10,
                    "Np_P50_STB": np_p50, "T_at_NP_P50_months": t_at_p50,
                    "Np_P90_STB": np_p90, "T_at_NP_P90_months": t_at_p90,
                    "Np_Base_STB": np_base_w, "T_at_NP_Base_months": t_base_w,
                })

            except Exception as e:
                rows.append({"well": w, "error": str(e)})

            prog.progress(i / len(wells_all))

        stat.success("✅ MC complete.")
        mc_all_df = pd.DataFrame(rows).round(4)
        st.dataframe(mc_all_df, use_container_width=True, height=520)

        out_buf = BytesIO()
        with pd.ExcelWriter(out_buf, engine="xlsxwriter") as wr:
            mc_all_df.to_excel(wr, sheet_name="MC_All", index=False)

        st.download_button("📥 Download Excel (All Wells P10/P50/P90 + Base)", out_buf.getvalue(),
                           file_name="all_wells_monte_carlo_p10_p50_p90_base.xlsx",
                           use_container_width=True)


