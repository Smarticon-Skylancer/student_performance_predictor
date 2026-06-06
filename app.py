import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
PRIMARY   = "#4F46E5"
SECONDARY = "#7C3AED"
SUCCESS   = "#059669"
WARNING   = "#D97706"
DANGER    = "#DC2626"
ACCENT    = "#0EA5E9"
GRADE_COLORS = {
    "A (85-100)": "#059669",
    "B (70-85)":  "#0EA5E9",
    "C (55-70)":  "#D97706",
    "D (40-55)":  "#F97316",
    "F (<40)":    "#DC2626",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #F8FAFC; }
    .block-container { padding: 1.5rem 2rem; }

    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size: 12px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #0F172A; line-height: 1; }
    .kpi-delta { font-size: 12px; margin-top: 4px; }
    .kpi-delta.up   { color: #059669; }
    .kpi-delta.down { color: #DC2626; }

    .section-header {
        font-size: 18px; font-weight: 600; color: #0F172A;
        margin: 1.5rem 0 .75rem; padding-bottom: .5rem;
        border-bottom: 2px solid #EEF2FF;
    }
    .insight-box {
        background: #EEF2FF; border-left: 4px solid #4F46E5;
        border-radius: 8px; padding: .75rem 1rem;
        font-size: 13px; color: #1E1B4B; line-height: 1.6; margin-bottom: .5rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; padding: 0 20px; border-radius: 8px;
        font-size: 13px; font-weight: 500;
        background: #F1F5F9; border: 1px solid #E2E8F0;
    }
    .stTabs [aria-selected="true"] {
        background: #4F46E5 !important; color: white !important;
    }
    div[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 1rem; border: 1px solid #E2E8F0; }
    .pred-score { font-size: 64px; font-weight: 800; text-align: center; line-height: 1; }
    .pred-grade { font-size: 22px; font-weight: 600; text-align: center; margin-top: 4px; }
    .stSlider > div > div > div > div { background: #4F46E5; }
    .stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 8rem;
    z-index: 999;
    background: #F8FAFC;
    padding: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/student_habits_performance.csv")
    df["parental_education_level"] = df["parental_education_level"].fillna(
        df["parental_education_level"].mode()[0]
    )
    bins   = [0, 40, 55, 70, 85, 100]
    labels = ["F (<40)", "D (40-55)", "C (55-70)", "B (70-85)", "A (85-100)"]
    df["grade"] = pd.cut(df["exam_score"], bins=bins, labels=labels)
    df["distraction_hours"] = df["social_media_hours"] + df["netflix_hours"]
    df["wellness_score"]    = (df["sleep_hours"] / 8 * 0.4 +
                                df["mental_health_rating"] / 10 * 0.4 +
                                df["exercise_frequency"] / 7 * 0.2) * 100
    return df

@st.cache_resource
def build_pipeline(df):
    ordinal_features = {
        "diet_quality":            ["Poor", "Fair", "Good"],
        "internet_quality":        ["Poor", "Average", "Good"],
        "parental_education_level":["High School", "Bachelor", "Master"],
    }
    nominal_features = ["gender", "part_time_job", "extracurricular_participation"]
    numeric_features = ["age","study_hours_per_day","social_media_hours","netflix_hours",
                        "attendance_percentage","sleep_hours","exercise_frequency","mental_health_rating"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("ord", OrdinalEncoder(
            categories=[ordinal_features[f] for f in ordinal_features],
            handle_unknown="use_encoded_value", unknown_value=-1),
         list(ordinal_features.keys())),
        ("nom", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
         nominal_features),
    ])
    data = df.drop(columns=["student_id","grade","distraction_hours","wellness_score"])
    X = data.drop(columns=["exam_score"])
    y = data["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

    models = {
        "Dummy Baseline":    Pipeline([("pre", preprocessor), ("model", DummyRegressor(strategy="mean"))]),
        "Linear Regression": Pipeline([("pre", preprocessor), ("model", LinearRegression())]),
        "Random Forest":     Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
        "Gradient Boosting": Pipeline([("pre", preprocessor), ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
        "Ridge (tuned)":     GridSearchCV(
            Pipeline([("pre", preprocessor), ("model", Ridge())]),
            {"model__alpha": [0.01,0.1,1,2,5,10,50,100]}, cv=5,
            scoring="neg_mean_squared_error", n_jobs=-1
        ),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = {
            "model": m, "pred": pred,
            "r2":   round(r2_score(y_test, pred), 4),
            "mae":  round(mean_absolute_error(y_test, pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, pred)), 4),
        }

    best_model  = results["Ridge (tuned)"]["model"]
    ridge_est   = best_model.best_estimator_ if hasattr(best_model, "best_estimator_") else best_model
    pre         = ridge_est["pre"]
    ohe_names   = pre.named_transformers_["nom"].get_feature_names_out(nominal_features).tolist()
    all_features = numeric_features + list(ordinal_features.keys()) + ohe_names
    coefs        = ridge_est["model"].coef_
    feat_df      = pd.DataFrame({"Feature": all_features, "Coefficient": coefs,
                                  "Abs": np.abs(coefs)}).sort_values("Abs", ascending=False)
    return results, best_model, X_train, X_test, y_train, y_test, feat_df

df   = load_data()
results, best_model, X_train, X_test, y_train, y_test, feat_df = build_pipeline(df)
best_pred = results["Ridge (tuned)"]["pred"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
         border-radius:14px;padding:1.2rem;margin-bottom:1rem;color:white;'>
      <div style='font-size:22px;font-weight:700;'>🎓 EduPredict</div>
      <div style='font-size:12px;opacity:.85;margin-top:4px;'>Student Performance Analytics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔍 Filters")
    gender_filter = st.multiselect("Gender", df["gender"].unique(), default=df["gender"].unique())
    grade_filter  = st.multiselect("Grade band", ["A (85-100)","B (70-85)","C (55-70)","D (40-55)","F (<40)"],
                                   default=["A (85-100)","B (70-85)","C (55-70)","D (40-55)","F (<40)"])
    job_filter    = st.multiselect("Part-time job", df["part_time_job"].unique(), default=df["part_time_job"].unique())
    study_range   = st.slider("Study hours/day", 0.0, 9.0, (0.0, 9.0), 0.5)
    score_range   = st.slider("Exam score range", 0, 100, (0, 100))

    filtered = df[
        df["gender"].isin(gender_filter) &
        df["grade"].isin(grade_filter) &
        df["part_time_job"].isin(job_filter) &
        df["study_hours_per_day"].between(*study_range) &
        df["exam_score"].between(*score_range)
    ]
    st.markdown(f"**{len(filtered):,}** of {len(df):,} students shown")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Analysis", "Model Performance", "Predict Score", "Data Explorer"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)

    def kpi(col, label, value, delta=None, delta_dir="up"):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {'<div class="kpi-delta '+delta_dir+'">'+delta+'</div>' if delta else ''}
        </div>""", unsafe_allow_html=True)

    avg_score = filtered["exam_score"].mean()
    pass_rate = (filtered["exam_score"] >= 55).mean() * 100
    top_pct   = (filtered["exam_score"] >= 85).mean() * 100
    avg_study = filtered["study_hours_per_day"].mean()
    avg_well  = filtered["wellness_score"].mean()

    kpi(k1, "Avg Exam Score",    f"{avg_score:.1f}",   f"σ = {filtered['exam_score'].std():.1f}")
    kpi(k2, "Pass Rate (≥55)",   f"{pass_rate:.1f}%",  f"{int(pass_rate/100*len(filtered))} students", "up")
    kpi(k3, "Top Performers (A)", f"{top_pct:.1f}%",   f"{int(top_pct/100*len(filtered))} students",   "up")
    kpi(k4, "Avg Study Hrs/Day", f"{avg_study:.1f}h",  f"Best: {filtered['study_hours_per_day'].max():.1f}h")
    kpi(k5, "Avg Wellness Score",f"{avg_well:.0f}/100", "Sleep+MH+Exercise")

    st.markdown("")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=filtered["exam_score"], nbinsx=35,
            marker_color=PRIMARY, opacity=0.85,
            name="All students",
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>"
        ))
        # Add normal curve overlay
        mu, sigma = filtered["exam_score"].mean(), filtered["exam_score"].std()
        x_norm = np.linspace(filtered["exam_score"].min(), filtered["exam_score"].max(), 200)
        y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(filtered) * (100/35)
        fig_hist.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines",
            line=dict(color=SECONDARY, width=2.5, dash="dot"), name="Normal fit"))
        for score, label, color in [(55,"Pass",WARNING),(85,"A Grade",SUCCESS)]:
            fig_hist.add_vline(x=score, line_dash="dash", line_color=color, line_width=1.5,
                               annotation_text=label, annotation_position="top right")
        fig_hist.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.1), plot_bgcolor="white",
            xaxis_title="Exam Score", yaxis_title="Students")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Grade Distribution</div>', unsafe_allow_html=True)
        grade_counts = filtered["grade"].value_counts().reindex(
            ["A (85-100)","B (70-85)","C (55-70)","D (40-55)","F (<40)"]).fillna(0)
        fig_pie = go.Figure(go.Pie(
            labels=grade_counts.index, values=grade_counts.values,
            hole=0.55, marker_colors=[GRADE_COLORS[g] for g in grade_counts.index],
            textinfo="percent+label",
            hovertemplate="%{label}<br>%{value} students (%{percent})<extra></extra>"
        ))
        fig_pie.add_annotation(text=f"<b>{len(filtered)}</b><br>students",
            x=0.5, y=0.5, showarrow=False, font_size=14)
        fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap: avg score by study hours x sleep hours
    st.markdown('<div class="section-header">Score Heatmap — Study Hours vs Sleep Hours</div>', unsafe_allow_html=True)
    filtered["study_bin"] = pd.cut(filtered["study_hours_per_day"],
        bins=[0,2,3,4,5,6,9], labels=["0-2h","2-3h","3-4h","4-5h","5-6h","6+h"])
    filtered["sleep_bin"] = pd.cut(filtered["sleep_hours"],
        bins=[3,5,6,7,8,12], labels=["<5h","5-6h","6-7h","7-8h","8+h"])
    heat_df = filtered.groupby(["study_bin","sleep_bin"], observed=True)["exam_score"].mean().unstack()
    fig_heat = px.imshow(heat_df, color_continuous_scale="RdYlGn",
        labels=dict(x="Sleep hours", y="Study hours/day", color="Avg Score"),
        text_auto=".1f", aspect="auto")
    fig_heat.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Quick insights
    st.markdown('<div class="section-header">💡 Auto Insights</div>', unsafe_allow_html=True)
    corr_study  = filtered["study_hours_per_day"].corr(filtered["exam_score"])
    corr_sm     = filtered["social_media_hours"].corr(filtered["exam_score"])
    corr_sleep  = filtered["sleep_hours"].corr(filtered["exam_score"])
    top_study   = filtered[filtered["grade"]=="A (85-100)"]["study_hours_per_day"].mean()
    bot_study   = filtered[filtered["grade"]=="F (<40)"]["study_hours_per_day"].mean()

    insights = [
        f"📚 Study hours have a <b>strong positive correlation</b> of <b>{corr_study:.2f}</b> with exam scores — the #1 driver.",
        f"📱 Social media usage correlates <b>{corr_sm:.2f}</b> with scores — more scrolling = lower performance.",
        f"😴 Sleep shows a correlation of <b>{corr_sleep:.2f}</b> — quality rest meaningfully boosts results.",
        f"🏆 A-grade students study <b>{top_study:.1f}h/day</b> on average vs <b>{bot_study:.1f}h/day</b> for F-grade — a {top_study-bot_study:.1f}h gap.",
        f"🎯 <b>{pass_rate:.1f}%</b> of the filtered cohort meets the 55-point pass threshold.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – DEEP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Feature Deep-Dive</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Scatter: study hours vs score coloured by grade
        fig_sc = px.scatter(filtered, x="study_hours_per_day", y="exam_score",
            color="grade", color_discrete_map=GRADE_COLORS,
            trendline="ols", trendline_scope="overall",
            labels={"study_hours_per_day":"Study hours/day","exam_score":"Exam score"},
            title="Study hours vs Exam score",
            hover_data=["gender","mental_health_rating"])
        fig_sc.update_traces(marker=dict(size=5, opacity=0.65))
        fig_sc.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0), plot_bgcolor="white")
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        # Box: score by diet quality
        order = ["Poor","Fair","Good"]
        fig_box = px.box(filtered, x="diet_quality", y="exam_score",
            color="diet_quality", category_orders={"diet_quality": order},
            color_discrete_sequence=[DANGER, WARNING, SUCCESS],
            title="Exam score by Diet quality",
            labels={"diet_quality":"Diet quality","exam_score":"Exam score"})
        fig_box.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Violin: mental health by grade
        fig_vio = px.violin(filtered, x="grade", y="mental_health_rating",
            color="grade", color_discrete_map=GRADE_COLORS, box=True,
            category_orders={"grade":["F (<40)","D (40-55)","C (55-70)","B (70-85)","A (85-100)"]},
            title="Mental health rating by grade",
            labels={"mental_health_rating":"Mental health (1-10)","grade":"Grade"})
        fig_vio.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    with c4:
        # Grouped bar: avg score by gender × part-time job
        gj = filtered.groupby(["gender","part_time_job"])["exam_score"].mean().reset_index()
        fig_gb = px.bar(gj, x="gender", y="exam_score", color="part_time_job",
            barmode="group", text_auto=".1f",
            color_discrete_sequence=[PRIMARY, ACCENT],
            title="Avg score: Gender × Part-time job",
            labels={"exam_score":"Avg exam score","part_time_job":"Part-time job"})
        fig_gb.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="white")
        st.plotly_chart(fig_gb, use_container_width=True)

    # Correlation matrix
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["study_hours_per_day","social_media_hours","netflix_hours",
                "attendance_percentage","sleep_hours","exercise_frequency",
                "mental_health_rating","distraction_hours","wellness_score","exam_score"]
    corr_mat = filtered[num_cols].corr()
    fig_corr = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=460, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Radar chart: avg profile by grade (A vs F)
    st.markdown('<div class="section-header">Student Profile Radar — A vs F grade</div>', unsafe_allow_html=True)
    radar_cols  = ["study_hours_per_day","attendance_percentage","sleep_hours",
                   "mental_health_rating","exercise_frequency"]
    radar_labels= ["Study hrs","Attendance %","Sleep hrs","Mental health","Exercise freq"]
    a_vals = filtered[filtered["grade"]=="A (85-100)"][radar_cols].mean()
    f_vals = filtered[filtered["grade"]=="F (<40)"][radar_cols].mean()

    # Normalise 0-1
    maxes = filtered[radar_cols].max()
    a_norm = (a_vals / maxes * 100).tolist()
    f_norm = (f_vals / maxes * 100).tolist()

    fig_radar = go.Figure()
    for vals, name, color in [(a_norm,"A Grade",SUCCESS),(f_norm,"F Grade",DANGER)]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name=name, line_color=color, fillcolor=color,
            opacity=0.25
        ))
    fig_radar.update_layout(height=380, polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        legend=dict(orientation="h", y=-0.1), margin=dict(l=40,r=40,t=30,b=40))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Distraction vs score bubble
    st.markdown('<div class="section-header">Distraction Hours vs Score (bubble = wellness)</div>', unsafe_allow_html=True)
    fig_bub = px.scatter(filtered.sample(min(400, len(filtered))),
        x="distraction_hours", y="exam_score",
        size="wellness_score", color="grade",
        color_discrete_map=GRADE_COLORS,
        labels={"distraction_hours":"Social media + Netflix hrs",
                "exam_score":"Exam score","wellness_score":"Wellness score"},
        hover_data=["study_hours_per_day","gender"])
    fig_bub.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white")
    st.plotly_chart(fig_bub, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Model Comparison</div>', unsafe_allow_html=True)

    # Metrics table
    metrics_rows = []
    for name, res in results.items():
        metrics_rows.append({"Model": name, "R²": res["r2"], "MAE": res["mae"], "RMSE": res["rmse"]})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("R²", ascending=False)

    # Horizontal bar chart for R²
    fig_r2 = go.Figure(go.Bar(
        x=metrics_df["R²"], y=metrics_df["Model"],
        orientation="h", text=[f"{v:.4f}" for v in metrics_df["R²"]],
        textposition="outside",
        marker_color=[SUCCESS if v > 0.85 else (WARNING if v > 0.5 else DANGER) for v in metrics_df["R²"]]
    ))
    fig_r2.update_layout(height=280, xaxis_range=[0,1.05], margin=dict(l=0,r=60,t=20,b=0),
        xaxis_title="R² Score", plot_bgcolor="white", title="R² Score by Model")
    st.plotly_chart(fig_r2, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        # Actual vs Predicted
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(x=y_test, y=best_pred, mode="markers",
            marker=dict(color=PRIMARY, opacity=0.5, size=5),
            name="Predictions",
            hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>"))
        perfect = [y_test.min(), y_test.max()]
        fig_ap.add_trace(go.Scatter(x=perfect, y=perfect, mode="lines",
            line=dict(color=DANGER, dash="dash", width=2), name="Perfect fit"))
        fig_ap.update_layout(height=340, title="Actual vs Predicted (Ridge tuned)",
            xaxis_title="Actual Score", yaxis_title="Predicted Score",
            plot_bgcolor="white", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_ap, use_container_width=True)

    with c2:
        # Residuals distribution
        residuals = y_test.values - best_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30,
            marker_color=SECONDARY, opacity=0.85, name="Residuals"))
        fig_res.add_vline(x=0, line_dash="dash", line_color=DANGER)
        fig_res.update_layout(height=340, title="Residuals Distribution",
            xaxis_title="Residual (Actual − Predicted)", yaxis_title="Count",
            plot_bgcolor="white", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (Ridge Coefficients)</div>', unsafe_allow_html=True)
    top_feats = feat_df.head(12)
    colors_feat = [SUCCESS if c > 0 else DANGER for c in top_feats["Coefficient"]]
    fig_fi = go.Figure(go.Bar(
        x=top_feats["Abs"], y=top_feats["Feature"],
        orientation="h", text=[f"{v:+.3f}" for v in top_feats["Coefficient"]],
        textposition="outside", marker_color=colors_feat
    ))
    fig_fi.update_layout(height=400, title="Top 12 features by absolute coefficient",
        xaxis_title="Absolute coefficient", plot_bgcolor="white",
        margin=dict(l=0,r=80,t=40,b=0))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Model metrics table
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df.style
        .background_gradient(subset=["R²"], cmap="Greens")
        .background_gradient(subset=["MAE","RMSE"], cmap="Reds_r")
        .format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}),
        use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICT SCORE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔮 Predict a Student\'s Exam Score</div>', unsafe_allow_html=True)
    st.markdown("Fill in a student's profile below and the model will predict their expected exam score.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Academic**")
        study_h     = st.slider("Study hours/day",  0.0, 9.0, 4.0, 0.1)
        attendance  = st.slider("Attendance %",      50.0, 100.0, 80.0, 0.5)
        internet_q  = st.selectbox("Internet quality", ["Poor","Average","Good"], index=1)
    with c2:
        st.markdown("**🧠 Lifestyle**")
        sleep_h     = st.slider("Sleep hours/night", 4.0, 10.0, 7.0, 0.5)
        sm_h        = st.slider("Social media hrs",  0.0, 8.0, 2.0, 0.5)
        netflix_h   = st.slider("Netflix hrs/day",   0.0, 6.0, 1.0, 0.5)
        exercise    = st.slider("Exercise days/week",0, 7, 3)
    with c3:
        st.markdown("**👤 Personal**")
        age         = st.slider("Age", 17, 24, 20)
        gender      = st.selectbox("Gender", ["Male","Female","Other"])
        diet_q      = st.selectbox("Diet quality", ["Poor","Fair","Good"], index=1)
        mental_h    = st.slider("Mental health (1-10)", 1, 10, 6)
        part_job    = st.selectbox("Part-time job", ["No","Yes"])
        parent_edu  = st.selectbox("Parent education", ["High School","Bachelor","Master"])
        extra       = st.selectbox("Extracurricular", ["No","Yes"])

    if st.button("🔮 Predict Score", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([{
            "age": age, "gender": gender,
            "study_hours_per_day": study_h, "social_media_hours": sm_h,
            "netflix_hours": netflix_h, "part_time_job": part_job,
            "attendance_percentage": attendance, "sleep_hours": sleep_h,
            "diet_quality": diet_q, "exercise_frequency": exercise,
            "parental_education_level": parent_edu, "internet_quality": internet_q,
            "mental_health_rating": mental_h, "extracurricular_participation": extra,
        }])

        pred_score = best_model.predict(input_df)[0]
        pred_score = max(0, min(100, pred_score))

        if pred_score >= 85:   grade_label, grade_color = "A — Excellent!", SUCCESS
        elif pred_score >= 70: grade_label, grade_color = "B — Good",        ACCENT
        elif pred_score >= 55: grade_label, grade_color = "C — Average",     WARNING
        elif pred_score >= 40: grade_label, grade_color = "D — Below avg",   "#F97316"
        else:                  grade_label, grade_color = "F — At risk",     DANGER

        st.markdown(f"""
        <div style='background:white;border-radius:16px;border:2px solid {grade_color};
             padding:2rem;text-align:center;margin:1rem 0;'>
          <div class="pred-score" style="color:{grade_color};">{pred_score:.1f}</div>
          <div class="pred-grade" style="color:{grade_color};">{grade_label}</div>
          <div style='font-size:13px;color:#64748B;margin-top:.5rem;'>out of 100 points</div>
        </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_score,
            delta={"reference": df["exam_score"].mean(), "suffix": " vs avg"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": grade_color},
                "steps": [
                    {"range": [0,40],   "color": "#FEE2E2"},
                    {"range": [40,55],  "color": "#FEF3C7"},
                    {"range": [55,70],  "color": "#FEF9C3"},
                    {"range": [70,85],  "color": "#DBEAFE"},
                    {"range": [85,100], "color": "#D1FAE5"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": .8, "value": df["exam_score"].mean()}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Personalised tips
        st.markdown('<div class="section-header">📝 Personalised Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if study_h < 4:
            tips.append(f"📚 Increase study time to at least 4h/day — currently {study_h}h. Top students average 5.6h.")
        if sm_h + netflix_h > 4:
            tips.append(f"📵 Reduce screen/distraction time. You have {sm_h+netflix_h:.1f}h/day — aim for under 3h.")
        if sleep_h < 7:
            tips.append(f"😴 Sleep more! {sleep_h}h is below the optimal 7-8h for cognitive performance.")
        if mental_h < 6:
            tips.append(f"🧘 Your mental health rating ({mental_h}/10) is low — consider stress management or counselling.")
        if attendance < 75:
            tips.append(f"🏫 Attendance ({attendance:.0f}%) is critically low. Regular attendance correlates strongly with performance.")
        if exercise < 3:
            tips.append(f"🏃 Exercise only {exercise}x/week — aim for 3-5 sessions. It boosts cognitive function.")
        if diet_q == "Poor":
            tips.append("🥗 Improve diet quality — poor nutrition impacts energy and concentration.")
        if not tips:
            tips.append("✅ Great profile! Keep maintaining these habits to stay in the top tier.")

        for t in tips:
            st.markdown(f'<div class="insight-box">{t}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📋 Raw Data Explorer</div>', unsafe_allow_html=True)

    col_pick = st.multiselect("Select columns to view",
        df.columns.tolist(), default=["student_id","exam_score","grade",
        "study_hours_per_day","sleep_hours","mental_health_rating","attendance_percentage"])
    sort_col = st.selectbox("Sort by", ["exam_score","study_hours_per_day","attendance_percentage"], index=0)
    sort_asc = st.checkbox("Ascending", value=False)

    display_df = filtered[col_pick].sort_values(sort_col, ascending=sort_asc) if sort_col in col_pick else filtered[col_pick]
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)
    st.caption(f"Showing {len(display_df):,} rows")

    st.markdown('<div class="section-header">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(filtered.describe().round(2), use_container_width=True)

    st.download_button("⬇ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="filtered_students.csv", mime="text/csv")
