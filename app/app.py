import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import warnings
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, mean_squared_error,
                             mean_absolute_error, r2_score)

warnings.filterwarnings('ignore')
shap.initjs()

st.set_page_config(page_title="ML Explorer", page_icon="🤖", layout="wide")

st.markdown("""
<style>

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #1a1a2e;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* ===== HEADERS ===== */
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    padding: 16px 0 8px;
    color: #1a1a2e;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f3460;
    border-left: 4px solid #e94560;
    padding-left: 10px;
    margin: 20px 0 10px;
}

/* ===== INFO BOX  ===== */
.info-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0;
    border-left: 5px solid #667eea;
    color: #1a1a2e !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.info-box h3,
.info-box p,
.info-box li {
    color: #1a1a2e;
}

/* ===== METRIC BOX ===== */
.metric-row {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #dee2e6;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🤖 ML Explorer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navighează",
    ["🏠 Acasă", "🫀 Clasificare", "🏡 Regresie"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Proiect ML · Sisteme Inteligente · 2026")

MODEL_DIR = "../models"
DATA_DIR = "../data"
RANDOM_STATE = 42

CLF_NEEDS_SCALING = ["logistic regression", "svm", "k-nearest neighbors", "naive bayes"]
REG_NEEDS_SCALING = ["linear regression", "svr", "knn regressor"]

def needs_sc(model_name, task="clf"):
    name = model_name.lower()
    lst = CLF_NEEDS_SCALING if task == "clf" else REG_NEEDS_SCALING
    return any(k in name for k in lst)


@st.cache_resource
def load_clf_data():
    models = {}
    if not os.path.exists(MODEL_DIR):
        return {}, None, []
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.startswith("clf_") and fname.endswith("_tuned.pkl"):
            raw = fname.replace("clf_", "").replace("_tuned.pkl", "")
            name = raw.replace("_", " ").title()
            models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
    scaler = joblib.load(os.path.join(MODEL_DIR, "clf_scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "clf_features.pkl"))
    return models, scaler, features


@st.cache_resource
def load_reg_data():
    models = {}
    if not os.path.exists(MODEL_DIR):
        return {}, None, []
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.startswith("reg_") and fname.endswith("_tuned.pkl"):
            raw = fname.replace("reg_", "").replace("_tuned.pkl", "")
            name = raw.replace("_", " ").title()
            models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
    scaler = joblib.load(os.path.join(MODEL_DIR, "reg_scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "reg_features.pkl"))
    return models, scaler, features


def get_shap_values(model, X_df, task="clf"):
    try:
        name = model.__class__.__name__.lower()

        if "linear" in name or "logistic" in name:
            explainer = shap.LinearExplainer(model, X_df)
            sv = explainer.shap_values(X_df)
            base = explainer.expected_value

            if isinstance(sv, list):
                sv = sv[1] if task == "clf" else sv[0]
                base = base[1] if hasattr(base, '__len__') else base

            return explainer, sv, base

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_df, check_additivity=False)
        base = explainer.expected_value

        if isinstance(sv, list):
            sv = sv[1]
            base = base[1] if hasattr(base, '__len__') else base

        return explainer, sv, base

    except Exception:
        explainer = shap.Explainer(model, X_df)
        obj = explainer(X_df)

        sv = obj.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        base = obj.base_values
        if hasattr(base, '__len__') and np.ndim(base) > 1:
            base = base[:, 1]

        return explainer, sv, base


def align_features(df_or_array, features, scaler=None, use_sc=False):

    if isinstance(df_or_array, pd.DataFrame):
        aligned = df_or_array.reindex(columns=features, fill_value=0)
    else:

        aligned = pd.DataFrame(df_or_array, columns=features)

    if use_sc and scaler is not None:
        return scaler.transform(aligned)
    return aligned.values


def build_model_comparison(models, X_plain, X_scaled, features, scaler, y_test, task="clf"):

    results = []
    for name, model in models.items():
        use = needs_sc(name, task)
        X_eval = align_features(X_plain, features, scaler, use_sc=use)
        y_pred = model.predict(X_eval)
        row = {"Model": name}
        if task == "clf":
            try:
                y_prob = model.predict_proba(X_eval)[:, 1]
            except Exception:
                y_prob = None
            row["Acuratețe"] = accuracy_score(y_test, y_pred)
            row["F1-Score"] = f1_score(y_test, y_pred)
            row["ROC-AUC"] = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        else:
            row["R2"] = r2_score(y_test, y_pred)
            row["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
            row["MAE"] = mean_absolute_error(y_test, y_pred)
        results.append(row)

    df = pd.DataFrame(results)
    sort_col = "ROC-AUC" if task == "clf" else "R2"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


# pagina acasa
if page == "🏠 Acasă":
    st.markdown('<div class="main-header">🤖 ML Explorer Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🫀 Clasificare — Boală Cardiacă</h3>
        <p>
        Prezice dacă un pacient prezintă risc de boală cardiacă pe baza
        caracteristicilor clinice: vârstă, colesterol, tensiune, etc.
        </p>
        <ul>
        <li>🎯 <b>Target:</b> 0 = Sănătos / 1 = Bolnav</li>
        <li>📊 <b>Dataset:</b> Heart Disease (UCI)</li>
        <li>🔢 <b>Features:</b> 13 caracteristici clinice</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🏡 Regresie — Prețuri Case</h3>
        <p>
        Estimează prețul de vânzare al unei locuințe pe baza
        caracteristicilor sale: suprafață, locație, condiție, etc.
        </p>
        <ul>
        <li>🎯 <b>Target:</b> Preț în USD</li>
        <li>📊 <b>Dataset:</b> King County House Sales</li>
        <li>🔢 <b>Features:</b> 18+ caracteristici imobiliare</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Modele utilizate")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Clasificare:**
        - Naïve Bayes
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - SVM / KNN / XGBoost / CatBoost / EBM
        """)
    with col2:
        st.markdown("""
        **Regresie:**
        - Linear Regression
        - Decision Tree / Random Forest
        - SVR / KNN
        - XGBoost / CatBoost / EBM
        """)


# pagina clasificare
elif page == "🫀 Clasificare":
    st.markdown('<div class="main-header">🫀 Clasificare — Boală Cardiacă</div>',
                unsafe_allow_html=True)

    try:
        df_clf = pd.read_csv(os.path.join(DATA_DIR, "heart_disease.csv"))
    except Exception:
        st.error("❌ Nu s-a găsit heart_disease.csv în folderul data/")
        st.stop()

    models_clf, scaler_clf, features_clf = load_clf_data()
    if not models_clf:
        st.error("❌ Nu există modele salvate în models/. Rulează mai întâi notebook-ul.")
        st.stop()

    with st.expander("📖 Descrierea Problemei și Datasetului", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Problema:** Prezicerea prezenței bolii cardiace la pacienți pe baza
            caracteristicilor clinice și demografice.

            **Variabila țintă:** `target` — 0 (sănătos) sau 1 (boală cardiacă prezentă)

            **Relevanță practică:** Boala cardiacă este una din principalele cauze
            de deces mondial. Un model ML poate ajuta medicii să identifice pacienții
            cu risc ridicat pentru intervenție timpurie.
            """)
        with col2:
            counts = df_clf['target'].value_counts()
            st.metric("Total pacienți", len(df_clf))
            st.metric("Sănătoși (0)", counts.get(0, 0))
            st.metric("Bolnavi (1)", counts.get(1, 0))

    st.markdown('<div class="section-header">📊 Explorarea Datelor (EDA)</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribuții", "Corelații", "Boxplot-uri"])

    with tab1:
        col1, col2 = st.columns(2)
        counts = df_clf['target'].value_counts()
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(['Sănătos (0)', 'Bolnav (1)'], counts.values,
                   color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
            ax.set_title('Distribuția Claselor', fontweight='bold')
            ax.set_ylabel('Număr observații')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(counts.values, labels=['Sănătos', 'Bolnav'],
                   colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05))
            ax.set_title('Proporția Claselor', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        num_cols = df_clf.select_dtypes(include='number').columns.drop('target')
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        for i, col in enumerate(num_cols[:12]):
            axes[i].hist(df_clf[df_clf['target'] == 0][col], alpha=0.6,
                         bins=20, label='Sănătos', color='#2ecc71')
            axes[i].hist(df_clf[df_clf['target'] == 1][col], alpha=0.6,
                         bins=20, label='Bolnav', color='#e74c3c')
            axes[i].set_title(col, fontweight='bold', fontsize=9)
            axes[i].legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Distribuții per Clasă', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        corr = df_clf.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, linewidths=0.3, ax=ax)
        ax.set_title('Matricea de Corelații', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.info("💡 `thalach` corelează negativ cu target — pacienții bolnavi au frecvență "
                "cardiacă maximă mai mică. `oldpeak` și `ca` corelează pozitiv cu boala.")

    with tab3:
        cont = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        for i, col in enumerate(cont):
            df_clf.boxplot(column=col, by='target', ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_xlabel('Target')
        plt.suptitle('Distribuție per Clasă', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    X_clf_full = df_clf.drop('target', axis=1)
    y_clf_full = df_clf['target']

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf_full, y_clf_full,
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_clf_full
    )
    sc_clf_local = StandardScaler()
    X_tr_c_sc = sc_clf_local.fit_transform(X_tr_c)
    X_te_c_sc = sc_clf_local.transform(X_te_c)

    st.markdown('<div class="section-header">🤖 Selectează Modelul</div>',
                unsafe_allow_html=True)

    model_name_clf = st.selectbox("Alege modelul de clasificare:",
                                  list(models_clf.keys()), key="clf_model")
    selected_clf = models_clf[model_name_clf]

    with st.expander("⚙️ Hiperparametrii modelului selectat"):
        params = selected_clf.get_params()
        params_df = pd.DataFrame(list(params.items()), columns=['Parametru', 'Valoare'])
        st.dataframe(params_df, use_container_width=True)

    st.markdown('<div class="section-header">📈 Performanța Modelului</div>',
                unsafe_allow_html=True)

    use_sc_c = needs_sc(model_name_clf, "clf")

    X_eval_c = align_features(X_te_c, features_clf, sc_clf_local, use_sc=use_sc_c)

    y_pred_c = selected_clf.predict(X_eval_c)
    y_prob_c = selected_clf.predict_proba(X_eval_c)[:, 1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acuratețe", f"{accuracy_score(y_te_c, y_pred_c):.4f}")
    col2.metric("Precizie", f"{precision_score(y_te_c, y_pred_c):.4f}")
    col3.metric("Recall", f"{recall_score(y_te_c, y_pred_c):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_te_c, y_pred_c):.4f}")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_te_c, y_prob_c):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_te_c, y_pred_c)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sănătos', 'Bolnav'],
                    yticklabels=['Sănătos', 'Bolnav'])
        ax.set_title(f'Matrice Confuzie — {model_name_clf}', fontweight='bold')
        ax.set_xlabel('Predicție');
        ax.set_ylabel('Real')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_te_c, y_prob_c)
        auc_val = roc_auc_score(y_te_c, y_prob_c)
        ax.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'AUC = {auc_val:.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate');
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Curbă ROC — {model_name_clf}', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    comp_df_c = build_model_comparison(
        models_clf,
        X_te_c,
        X_te_c_sc,
        features_clf,
        sc_clf_local,
        y_te_c,
        task="clf"
    )
    st.markdown('<div class="section-header">📊 Comparare Toate Modelele</div>',
                unsafe_allow_html=True)
    st.dataframe(
        comp_df_c.style.background_gradient(
            cmap='RdYlGn',
            subset=[c for c in comp_df_c.columns if c != "Model"]
        ),
        use_container_width=True
    )

    st.markdown('<div class="section-header">📉 Curbele de Învățare</div>',
                unsafe_allow_html=True)

    with st.spinner("Calculez curbele de învățare..."):

        X_lc_c = align_features(X_tr_c, features_clf, sc_clf_local, use_sc=use_sc_c)
        train_sizes, train_sc_lc, val_sc_lc = learning_curve(
            selected_clf, X_lc_c, y_tr_c,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring='roc_auc'
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_sizes, train_sc_lc.mean(axis=1), 'o-', color='#3498db',
                label='Train', linewidth=2)
        ax.fill_between(train_sizes,
                        train_sc_lc.mean(axis=1) - train_sc_lc.std(axis=1),
                        train_sc_lc.mean(axis=1) + train_sc_lc.std(axis=1),
                        alpha=0.15, color='#3498db')
        ax.plot(train_sizes, val_sc_lc.mean(axis=1), 'o-', color='#e74c3c',
                label='Validare', linewidth=2)
        ax.fill_between(train_sizes,
                        val_sc_lc.mean(axis=1) - val_sc_lc.std(axis=1),
                        val_sc_lc.mean(axis=1) + val_sc_lc.std(axis=1),
                        alpha=0.15, color='#e74c3c')
        gap_c = train_sc_lc.mean(axis=1)[-1] - val_sc_lc.mean(axis=1)[-1]
        status_c = "⚠️ Posibil Overfitting" if gap_c > 0.05 else "✅ Model bine generalizat"
        ax.set_title(f'Learning Curve — {model_name_clf}\n{status_c} (gap={gap_c:.3f})',
                     fontweight='bold')
        ax.set_xlabel('Nr. observații antrenare')
        ax.set_ylabel('ROC-AUC')
        ax.legend();
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Predicție Interactivă</div>',
                unsafe_allow_html=True)
    st.markdown("Introdu valorile caracteristicilor pentru a obține o predicție:")

    input_clf = {}
    cols_input = st.columns(4)
    for i, feat in enumerate(features_clf):
        with cols_input[i % 4]:
            mn = float(df_clf[feat].min())
            mx = float(df_clf[feat].max())
            mv = float(df_clf[feat].mean())
            step = 1.0 if df_clf[feat].dtype in ['int64', 'int32'] else 0.1
            input_clf[feat] = st.number_input(feat, mn, mx, mv, step, key=f"clf_{feat}")

    if st.button("🚀 Prezice!", type="primary", key="clf_predict"):
        X_in = pd.DataFrame([input_clf])

        X_proc_c = align_features(X_in, features_clf, scaler_clf, use_sc=needs_sc(model_name_clf, "clf"))

        pred = selected_clf.predict(X_proc_c)[0]
        prob = selected_clf.predict_proba(X_proc_c)[0]

        st.markdown("### Rezultat:")
        if pred == 1:
            st.error(f"❌ **Risc de Boală Cardiacă Detectat**\n\nProbabilitate boală: **{prob[1]:.1%}**")
        else:
            st.success(f"✅ **Pacient Sănătos**\n\nProbabilitate sănătos: **{prob[0]:.1%}**")

        st.markdown("#### 🔍 Explicabilitate SHAP — ce a influențat predicția")
        try:
            X_bg_raw = X_clf_full.sample(min(100, len(X_clf_full)), random_state=42)
            X_bg = pd.DataFrame(
                align_features(X_bg_raw, features_clf, scaler_clf,
                               use_sc=needs_sc(model_name_clf, "clf")),
                columns=features_clf
            )

            X_in_df = pd.DataFrame(X_proc_c, columns=features_clf)

            _, sv_pred, base_val = get_shap_values(selected_clf, X_bg, "clf")

            explainer = shap.Explainer(selected_clf, X_bg)
            sv_single = explainer(X_in_df).values

            if sv_single.ndim == 3:
                sv_single = sv_single[:, :, 1]

            sv_s = sv_single[0]

            exp = shap.Explanation(
                values=sv_s,
                base_values=base_val,
                data=X_in_df.iloc[0].values,
                feature_names=features_clf
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(exp, show=False)
            st.pyplot(fig)
            plt.close()

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv_s]
                idx_sort = np.argsort(np.abs(sv_s))[::-1][:10]

                ax.barh([features_clf[j] for j in idx_sort[::-1]],
                        sv_s[idx_sort[::-1]],
                        color=[colors[j] for j in idx_sort[::-1]])

                ax.axvline(0, color='black')
                ax.set_title('Impact SHAP (local)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                mean_abs = np.abs(sv_pred).mean(axis=0)
                idx = np.argsort(mean_abs)[::-1][:10]

                ax.barh([features_clf[j] for j in idx[::-1]],
                        mean_abs[idx[::-1]],
                        color='#3498db')

                ax.set_title('Importanță globală (|SHAP|)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            top3 = [features_clf[j] for j in np.argsort(np.abs(sv_s))[::-1][:3]]
            st.info(f"🏆 Top 3 caracteristici: {', '.join(top3)}")

        except Exception as e:
            st.warning(f"SHAP indisponibil: {e}")


# pagina regresie
elif page == "🏡 Regresie":
    st.markdown('<div class="main-header">🏡 Regresie — Prețuri Locuințe</div>',
                unsafe_allow_html=True)

    try:
        df_raw = pd.read_csv(os.path.join(DATA_DIR, "house_prices.csv"))
    except Exception:
        st.error("❌ Nu s-a găsit house_prices.csv în folderul data/")
        st.stop()

    df_reg = df_raw.copy()
    for col_drop in ['date', 'id']:
        if col_drop in df_reg.columns:
            df_reg = df_reg.drop(col_drop, axis=1)

    df_reg['age'] = 2026 - df_reg['yr_built']

    obj_cols = df_reg.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df_reg = df_reg.drop(columns=obj_cols)
    df_reg = df_reg.dropna()

    X_reg_full = df_reg.drop('price', axis=1)
    y_reg_full = df_reg['price']

    models_reg, scaler_reg, features_reg = load_reg_data()
    if not models_reg:
        st.error("❌ Nu există modele salvate. Rulează mai întâi notebook-ul.")
        st.stop()

    with st.expander("📖 Descrierea Problemei și Datasetului", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Problema:** Predicția prețului de vânzare al unei locuințe pe baza
            caracteristicilor sale fizice și a localizării geografice.

            **Variabila țintă:** `price` — prețul în USD

            **Relevanță practică:** Estimarea corectă a valorii imobiliare
            ajută cumpărătorii, vânzătorii și agențiile imobiliare în luarea
            deciziilor financiare.
            """)
        with col2:
            st.metric("Total observații", len(df_reg))
            st.metric("Preț mediu", f"${y_reg_full.mean():,.0f}")
            st.metric("Interval preț", f"${y_reg_full.min():,} – ${y_reg_full.max():,}")

    st.markdown('<div class="section-header">📊 Explorarea Datelor (EDA)</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribuții", "Corelații", "Scatter"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df_reg['price'], bins=50, color='#3498db', edgecolor='white')
            ax.set_title('Distribuția Prețurilor', fontweight='bold')
            ax.set_xlabel('Preț (USD)')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(np.log1p(df_reg['price']), bins=50, color='#2ecc71', edgecolor='white')
            ax.set_title('Distribuția log(Preț)', fontweight='bold')
            ax.set_xlabel('log(Preț)')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_r = df_reg.corr(numeric_only=True)
        mask_r = np.triu(np.ones_like(corr_r, dtype=bool))
        sns.heatmap(corr_r, mask=mask_r, cmap='coolwarm', center=0,
                    linewidths=0.3, ax=ax, annot=False)
        ax.set_title('Matricea de Corelații', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.info("💡 `sqft_living` și `grade` au cea mai mare corelație cu prețul.")

    with tab3:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, col in zip(axes, ['sqft_living', 'grade', 'age']):
            ax.scatter(df_reg[col], df_reg['price'], s=5, alpha=0.3, color='#3498db')
            ax.set_xlabel(col);
            ax.set_ylabel('Preț')
            ax.set_title(f'{col} vs Preț', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    X_reg_aligned = X_reg_full.reindex(columns=features_reg, fill_value=0)

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg_aligned, y_reg_full,
        test_size=0.25, random_state=RANDOM_STATE
    )
    sc_r = StandardScaler()
    X_tr_r_sc = sc_r.fit_transform(X_tr_r)
    X_te_r_sc = sc_r.transform(X_te_r)

    st.markdown('<div class="section-header">🤖 Selectează Modelul</div>',
                unsafe_allow_html=True)

    model_name_reg = st.selectbox("Alege modelul de regresie:",
                                  list(models_reg.keys()), key="reg_model")
    selected_reg = models_reg[model_name_reg]

    with st.expander("⚙️ Hiperparametrii modelului selectat"):
        params_r = selected_reg.get_params()
        params_df_r = pd.DataFrame(list(params_r.items()), columns=['Parametru', 'Valoare'])
        st.dataframe(params_df_r, use_container_width=True)

    st.markdown('<div class="section-header">📈 Performanța Modelului</div>',
                unsafe_allow_html=True)

    use_sc_r = needs_sc(model_name_reg, "reg")
    X_eval_r = X_te_r_sc if use_sc_r else X_te_r.values

    y_pred_r = selected_reg.predict(X_eval_r)
    mse_r = mean_squared_error(y_te_r, y_pred_r)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2_score(y_te_r, y_pred_r):.4f}")
    col2.metric("RMSE", f"${np.sqrt(mse_r):,.0f}")
    col3.metric("MAE", f"${mean_absolute_error(y_te_r, y_pred_r):,.0f}")
    col4.metric("MSE", f"{mse_r:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te_r, y_pred_r, s=5, alpha=0.3, color='#3498db')
        mn_r, mx_r = y_te_r.min(), y_te_r.max()
        ax.plot([mn_r, mx_r], [mn_r, mx_r], 'r--', linewidth=1.5, label='Perfect')
        ax.set_xlabel('Valori Reale');
        ax.set_ylabel('Predicții')
        ax.set_title(f'Predicție vs Real — {model_name_reg}', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        residuals = y_te_r - y_pred_r
        ax.hist(residuals, bins=40, color='#9b59b6', edgecolor='white')
        ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Rezidual (Real - Prezis)')
        ax.set_title('Distribuția Reziduarilor', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    comp_df_r = build_model_comparison(
        models_reg, X_te_r, X_te_r_sc, features_reg, sc_r, y_te_r, task="reg"
    )
    st.markdown('<div class="section-header">📊 Comparare Toate Modelele</div>',
                unsafe_allow_html=True)
    st.dataframe(
        comp_df_r.style.background_gradient(
            cmap='RdYlGn',
            subset=[c for c in comp_df_r.columns if c != "Model"]
        ),
        use_container_width=True
    )

    st.markdown('<div class="section-header">📉 Curbele de Învățare</div>',
                unsafe_allow_html=True)

    with st.spinner("Calculez curbele de învățare..."):
        X_lc_r = X_tr_r_sc if use_sc_r else X_tr_r.values
        tr_s_r, tr_sc_r, val_sc_r = learning_curve(
            selected_reg, X_lc_r, y_tr_r,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring='r2'
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tr_s_r, tr_sc_r.mean(axis=1), 'o-', color='#3498db', label='Train', linewidth=2)
        ax.fill_between(tr_s_r,
                        tr_sc_r.mean(axis=1) - tr_sc_r.std(axis=1),
                        tr_sc_r.mean(axis=1) + tr_sc_r.std(axis=1),
                        alpha=0.15, color='#3498db')
        ax.plot(tr_s_r, val_sc_r.mean(axis=1), 'o-', color='#e74c3c', label='Validare', linewidth=2)
        ax.fill_between(tr_s_r,
                        val_sc_r.mean(axis=1) - val_sc_r.std(axis=1),
                        val_sc_r.mean(axis=1) + val_sc_r.std(axis=1),
                        alpha=0.15, color='#e74c3c')
        gap_r = tr_sc_r.mean(axis=1)[-1] - val_sc_r.mean(axis=1)[-1]
        status_r = "⚠️ Posibil Overfitting" if gap_r > 0.05 else "✅ Bun"
        ax.set_title(f'Learning Curve — {model_name_reg}\n{status_r} (gap={gap_r:.3f})',
                     fontweight='bold')
        ax.set_xlabel('Nr. observații antrenare')
        ax.set_ylabel('R²');
        ax.legend();
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Predicție Interactivă</div>',
                unsafe_allow_html=True)

    input_reg = {}
    cols_r = st.columns(4)
    for i, feat in enumerate(features_reg):
        with cols_r[i % 4]:
            if feat in X_reg_full.columns:
                mn_f = float(X_reg_full[feat].min())
                mx_f = float(X_reg_full[feat].max())
                mv_f = float(X_reg_full[feat].mean())
                step_f = 1.0 if X_reg_full[feat].dtype in ['int64', 'int32'] else 0.1
            else:
                mn_f, mx_f, mv_f, step_f = 0.0, 100.0, 10.0, 1.0
            input_reg[feat] = st.number_input(feat, mn_f, mx_f, mv_f, step_f,
                                              key=f"reg_{feat}")

    if st.button("💰 Estimează Prețul!", type="primary", key="reg_predict"):
        X_in_r = pd.DataFrame([input_reg])
        X_proc_r = align_features(X_in_r, features_reg, scaler_reg,
                                  use_sc=needs_sc(model_name_reg, "reg"))
        pred_price = selected_reg.predict(X_proc_r)[0]
        st.success(f"## 💰 Preț Estimat: **${pred_price:,.0f}**")

        st.markdown("#### 🔍 Explicabilitate SHAP")
        try:
            X_bg_raw = X_reg_full.sample(min(100, len(X_reg_full)), random_state=42)
            X_bg = pd.DataFrame(
                align_features(X_bg_raw, features_reg, scaler_reg,
                               use_sc=needs_sc(model_name_reg, "reg")),
                columns=features_reg
            )
            X_in_df = pd.DataFrame(X_proc_r, columns=features_reg)
            _, sv_bg, base_val = get_shap_values(selected_reg, X_bg, "reg")

            explainer = shap.Explainer(selected_reg, X_bg)
            sv_single = explainer(X_in_df).values
            sv_s = sv_single[0]

            exp = shap.Explanation(
                values=sv_s,
                base_values=base_val,
                data=X_in_df.iloc[0].values,
                feature_names=features_reg
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(exp, show=False)
            st.pyplot(fig)
            plt.close()

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv_s]
                idx = np.argsort(np.abs(sv_s))[::-1][:10]
                ax.barh([features_reg[j] for j in idx[::-1]],
                        sv_s[idx[::-1]],
                        color=[colors[j] for j in idx[::-1]])
                ax.axvline(0, color='black')
                ax.set_title('Impact SHAP (local)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                mean_abs = np.abs(sv_bg).mean(axis=0)
                idx2 = np.argsort(mean_abs)[::-1][:10]
                ax.barh([features_reg[j] for j in idx2[::-1]],
                        mean_abs[idx2[::-1]],
                        color='#f39c12')
                ax.set_title('Importanță globală (|SHAP|)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            top3 = [features_reg[j] for j in np.argsort(np.abs(sv_s))[::-1][:3]]
            st.info(f"🏆 Top 3 caracteristici: {', '.join(top3)}")

        except Exception as e:
            st.warning(f"SHAP indisponibil: {e}")