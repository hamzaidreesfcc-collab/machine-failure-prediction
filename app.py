"""
Machine Failure Prediction System — Streamlit App
Capstone Project 01
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os

# ─── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictML · Machine Failure System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #F1EFE8; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─── Load saved models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        lr_model      = joblib.load('lr_model.pkl')
        knn_model     = joblib.load('knn_model.pkl')
        scaler        = joblib.load('scaler.pkl')
        le            = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return lr_model, knn_model, scaler, le, feature_names, True
    except FileNotFoundError as e:
        return None, None, None, None, None, False

lr_model, knn_model, scaler, le, feature_names, models_loaded = load_models()

if not models_loaded:
    st.error("❌ Model files not found! Make sure these files are in the same folder as app.py:\n\n"
             "lr_model.pkl · knn_model.pkl · scaler.pkl · label_encoder.pkl · feature_names.pkl")
    st.stop()

# ─── Load dataset for stats ───────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('predictive_maintenance.csv')
        return df, True
    except FileNotFoundError:
        return None, False

df, data_loaded = load_data()

# ─── Prediction function ──────────────────────────────────────────
def predict(machine_type, air_temp, proc_temp, rpm, torque, tool_wear, model_choice):
    type_enc  = le.transform([machine_type])[0]
    temp_diff = proc_temp - air_temp
    power     = torque * rpm / 9549

    features = pd.DataFrame([{
        'Type'                    : type_enc,
        'Air temperature [K]'     : air_temp,
        'Process temperature [K]' : proc_temp,
        'Rotational speed [rpm]'  : rpm,
        'Torque [Nm]'             : torque,
        'Tool wear [min]'         : tool_wear,
        'Temp_diff'               : temp_diff,
        'Power'                   : power
    }])[feature_names]

    scaled = scaler.transform(features)
    model  = knn_model if model_choice == 'KNN' else lr_model
    prob   = model.predict_proba(scaled)[0][1]
    pred   = model.predict(scaled)[0]
    return prob * 100, int(pred)


# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ PredictML")
    st.markdown("*Predictive Maintenance System*")
    st.markdown("---")
    page = st.radio("Navigate", ["Dashboard", "Predict Failure", "Sensor Charts", "Model Info"])
    st.markdown("---")
    model_choice = st.selectbox("Active Model", ["KNN", "Logistic Regression"])
    st.markdown("---")
    st.success("✅ Models loaded from .pkl files")
    st.caption("Capstone Project 01")


# ══════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("⚙️ Machine Health Dashboard")
    st.caption("Predictive Maintenance · Real Dataset · 10,000 records")

    if data_loaded:
        total      = len(df)
        failures   = df['Target'].sum()
        fail_rate  = failures / total * 100
        avg_wear   = df['Tool wear [min]'].mean()
        fail_types = df['Failure Type'].value_counts()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",  f"{total:,}")
        c2.metric("Total Failures", f"{int(failures):,}")
        c3.metric("Failure Rate",   f"{fail_rate:.1f}%")
        c4.metric("Avg Tool Wear",  f"{avg_wear:.0f} min")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Failure Rate by Machine Type")
            type_failure = df.groupby('Type')['Target'].mean().mul(100).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#E24B4A', '#BA7517', '#378ADD']
            bars = ax.bar(type_failure.index, type_failure.values,
                          color=colors[:len(type_failure)], edgecolor='white', width=0.5)
            for bar, val in zip(bars, type_failure.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
            ax.set_ylabel('Failure Rate (%)')
            ax.set_xlabel('Machine Type  (H=High, L=Low, M=Medium)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.subheader("Failure Type Breakdown")
            fig, ax = plt.subplots(figsize=(5, 3))
            colors2 = ['#1D9E75','#E24B4A','#BA7517','#378ADD','#534AB7','#D85A30']
            bars2 = ax.barh(fail_types.index, fail_types.values,
                            color=colors2[:len(fail_types)], edgecolor='white')
            for bar, val in zip(bars2, fail_types.values):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontweight='bold', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("Dataset Sample")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("CSV file not found. Place predictive_maintenance.csv in the same folder.")


# ══════════════════════════════════════════════════════════════════
# PAGE: Predict Failure
# ══════════════════════════════════════════════════════════════════
elif page == "Predict Failure":
    st.title("🔍 Predict Machine Failure")
    st.caption(f"Active model: **{model_choice}**")

    st.subheader("Enter Sensor Readings")
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
                                    help="L = Low grade · M = Medium · H = High")
        air_temp     = st.slider("Air Temperature (K)",     295.0, 310.0, 298.5, 0.1)
        proc_temp    = st.slider("Process Temperature (K)", 305.0, 320.0, 308.5, 0.1)
        rpm          = st.slider("Rotational Speed (RPM)",  1000,  3000,  1500)

    with col2:
        torque    = st.slider("Torque (Nm)",     5.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("Tool Wear (min)", 0,   250,  100)
        st.markdown("&nbsp;")
        st.info(f"**Temp Difference:** {proc_temp - air_temp:.1f} K\n\n"
                f"**Power (approx):** {torque * rpm / 9549:.2f} kW")

    st.markdown("---")

    if st.button("⚡ Predict Failure Probability", type="primary", use_container_width=True):
        prob, pred = predict(machine_type, air_temp, proc_temp,
                             rpm, torque, tool_wear, model_choice)

        risk = "Critical" if prob > 60 else "High" if prob > 40 else "Moderate" if prob > 20 else "Low"
        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Probability", f"{prob:.1f}%")
        c2.metric("Prediction",          "⚠️ FAILURE" if pred == 1 else "✅ NORMAL")
        c3.metric("Risk Level",          risk)

        # Gauge bar
        st.markdown("#### Risk Gauge")
        fig, ax = plt.subplots(figsize=(10, 1.2))
        ax.barh([0], [100], color='#EEEEEE', height=0.5)
        bar_color = "#E24B4A" if prob > 60 else "#BA7517" if prob > 30 else "#1D9E75"
        ax.barh([0], [prob], color=bar_color, height=0.5)
        ax.axvline(prob, color='#333', linewidth=2)
        ax.axvline(30, color='#BA7517', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(60, color='#E24B4A', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(15, 0.4,  'Low',      ha='center', fontsize=9, color='#1D9E75')
        ax.text(45, 0.4,  'Moderate', ha='center', fontsize=9, color='#BA7517')
        ax.text(80, 0.4,  'Critical', ha='center', fontsize=9, color='#E24B4A')
        ax.text(prob, -0.38, f'{prob:.1f}%', ha='center', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Failure Probability (%)")
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        if pred == 1:
            st.error("🚨 **Maintenance Required!** Schedule immediate inspection.")
        elif prob > 30:
            st.warning("⚠️ **Watch Mode.** Plan preventive maintenance soon.")
        else:
            st.success("✅ **Machine operating normally.** Continue regular monitoring.")


# ══════════════════════════════════════════════════════════════════
# PAGE: Sensor Charts
# ══════════════════════════════════════════════════════════════════
elif page == "Sensor Charts":
    st.title("📊 Sensor Data Charts")

    if data_loaded:
        machine_ids = df['Product ID'].unique()[:20]
        selected    = st.selectbox("Select Machine (Product ID)", machine_ids)
        machine_df  = df[df['Product ID'] == selected]

        st.markdown(f"**Showing data for:** `{selected}` — "
                    f"Type: `{machine_df['Type'].values[0]}` — "
                    f"Status: {'⚠️ FAILED' if machine_df['Target'].values[0]==1 else '✅ Normal'}")
        st.markdown("---")

        numeric_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        thresholds = {
            'Air temperature [K]'     : None,
            'Process temperature [K]' : 309,
            'Rotational speed [rpm]'  : 1200,
            'Torque [Nm]'             : None,
            'Tool wear [min]'         : 220
        }

        cols = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(df[df['Target']==0][col], bins=40, alpha=0.6,
                        color='#1D9E75', label='No Failure')
                ax.hist(df[df['Target']==1][col], bins=40, alpha=0.6,
                        color='#E24B4A', label='Failure')
                val = machine_df[col].values[0]
                ax.axvline(val, color='#2C2C2A', linewidth=2,
                           linestyle='--', label=f'Selected: {val}')
                if thresholds[col]:
                    ax.axvline(thresholds[col], color='orange', linewidth=1,
                               linestyle=':', label=f'Threshold: {thresholds[col]}')
                ax.set_title(col, fontweight='bold', fontsize=10)
                ax.legend(fontsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    else:
        st.warning("CSV file not found. Place predictive_maintenance.csv in the same folder.")


# ══════════════════════════════════════════════════════════════════
# PAGE: Model Info
# ══════════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.title("🧠 Model Information")

    if data_loaded:
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

        df_ml = df.drop(columns=['UDI', 'Product ID', 'Failure Type'])
        df_ml['Type']      = le.transform(df_ml['Type'])
        df_ml['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        df_ml['Power']     = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 9549

        X = df_ml.drop(columns=['Target'])[feature_names]
        y = df_ml['Target']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_s = scaler.transform(X_test)

        lr_pred   = lr_model.predict(X_test_s)
        knn_pred  = knn_model.predict(X_test_s)
        lr_proba  = lr_model.predict_proba(X_test_s)[:, 1]
        knn_proba = knn_model.predict_proba(X_test_s)[:, 1]
        lr_acc    = accuracy_score(y_test, lr_pred)
        knn_acc   = accuracy_score(y_test, knn_pred)
        lr_auc    = roc_auc_score(y_test, lr_proba)
        knn_auc   = roc_auc_score(y_test, knn_proba)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Logistic Regression")
            st.metric("Test Accuracy", f"{lr_acc*100:.2f}%")
            st.metric("ROC-AUC",       f"{lr_auc:.4f}")
        with col_r:
            st.subheader("K-Nearest Neighbors")
            st.metric("Test Accuracy", f"{knn_acc*100:.2f}%")
            st.metric("ROC-AUC",       f"{knn_auc:.4f}")

        st.markdown("---")
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, preds, name, cmap, acc in zip(
            axes,
            [lr_pred, knn_pred],
            ['Logistic Regression', 'KNN'],
            ['Blues', 'Greens'],
            [lr_acc, knn_acc]
        ):
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                        xticklabels=['No Fail','Fail'],
                        yticklabels=['No Fail','Fail'],
                        linewidths=1, linecolor='white',
                        annot_kws={'size': 13, 'weight': 'bold'})
            ax.set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("ROC Curves")
        fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba)
        fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr_lr,  tpr_lr,  lw=2, color='#378ADD',
                label=f'Logistic Reg  (AUC={lr_auc:.3f})')
        ax.plot(fpr_knn, tpr_knn, lw=2, color='#1D9E75',
                label=f'KNN  (AUC={knn_auc:.3f})')
        ax.plot([0,1],[0,1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison', fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Feature Importance (LR Coefficients)")
        coef       = lr_model.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        colors     = ['#E24B4A' if c > 0 else '#1D9E75' for c in coef]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([feature_names[i] for i in sorted_idx],
               [coef[i] for i in sorted_idx],
               color=[colors[i] for i in sorted_idx], edgecolor='white')
        ax.axhline(0, color='#888', linewidth=0.8)
        ax.set_title('Feature Coefficients  (red = increases failure risk)', fontweight='bold')
        ax.set_ylabel('Coefficient')
        plt.xticks(rotation=20, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("CSV file not found. Place predictive_maintenance.csv in the same folder.")

# ─── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption("PredictML · Machine Failure Prediction System · Capstone Project 01")
