import os
import pickle
import numpy as np
import pandas as pd
import markdown
import streamlit as st
from openai import OpenAI
from streamlit_option_menu import option_menu

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI — Disease Prediction System",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,700;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-family:'Fraunces',serif; font-size:2.5rem; color:#0f172a; line-height:1.15; margin-bottom:0.25rem; }
.main-title span { color:#0891b2; }
.subtitle { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.kpi-card { background:#fff; border:1px solid #e2e8f0; border-top:3px solid #0891b2; border-radius:10px; padding:1rem 1.2rem; text-align:center; }
.kpi-val { font-size:1.9rem; font-weight:700; color:#0f172a; }
.kpi-lbl { font-size:0.75rem; color:#64748b; margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.05em; }
.result-pos { background:#fff1f2; border-left:4px solid #e11d48; border-radius:8px; padding:1rem 1.4rem; color:#9f1239; font-weight:600; margin-top:1rem; }
.result-neg { background:#f0fdf4; border-left:4px solid #16a34a; border-radius:8px; padding:1rem 1.4rem; color:#15803d; font-weight:600; margin-top:1rem; }
.chat-user { background:#0891b2; color:white; border-radius:16px 16px 4px 16px; padding:0.7rem 1rem; margin:0.4rem 0; margin-left:25%; font-size:0.9rem; }
.chat-bot { background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px 16px 16px 4px; padding:0.7rem 1rem; margin:0.4rem 0; margin-right:25%; font-size:0.9rem; color:#1e293b; }
.info-box { background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; padding:0.9rem 1.1rem; color:#0c4a6e; font-size:0.88rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load all pickle models ────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    files = {
        'diabetes':      'saved_models/diabetes_model.sav',
        'heart':         'saved_models/heart_disease_model.sav',
        'parkinsons':    'saved_models/parkinsons_model.sav',
        'breast_cancer': 'saved_models/breast_cancer_model.sav',
        'kidney':        'saved_models/kidney_disease_model.sav',
        'liver':         'saved_models/liver_disease_model.sav',
    }
    loaded = {}
    for key, rel_path in files.items():
        full = os.path.join(working_dir, rel_path)
        if os.path.exists(full):
            try:
                loaded[key] = pickle.load(open(full, 'rb'))
            except Exception:
                pass
    return loaded

@st.cache_resource
def load_brain_model():
    from keras.models import load_model as keras_load
    path = os.path.join(working_dir, 'saved_models/brain_tumor_model.keras')
    if os.path.exists(path):
        return keras_load(path)
    return None

models = load_all_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:2rem;font-weight:bold;'>🧬 MedAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.9rem;margin-top:-15px;margin-left:8px;'>Multiple Disease Prediction</p>", unsafe_allow_html=True)
    st.markdown("---")
    selected = option_menu(
        menu_title=None,
        options=['Dashboard','Brain Tumour','Diabetes','Heart Disease','Breast Cancer','Kidney Disease'
                 ,'Liver Disease',"Parkinson's",'Health Chatbot'],
        icons=['bar-chart-line-fill','file-medical','droplet-half','heart-pulse','gender-female','funnel','capsule','person-arms-up','robot'],
        default_index=0,
    )
    st.markdown("---")
    st.markdown("<p style='color:#475569;font-size:0.75rem;text-align:center'>⚕️ For educational use only.<br>Consult a doctor for diagnosis.</p>", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def show_result(positive, pos_msg, neg_msg):
    if positive:
        st.markdown(f'<div class="result-pos">⚠️ {pos_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-neg">✅ {neg_msg}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Dashboard':
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<p class="main-title">Health Analytics <span>Dashboard</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visual insights from real medical datasets — patterns, risk factors & global disease burden.</p>', unsafe_allow_html=True)

    @st.cache_data
    def get_datasets():
        from sklearn.datasets import load_breast_cancer

        def try_load(local_name, url):
            local = os.path.join(working_dir, local_name)
            if os.path.exists(local):
                return pd.read_csv(local)
            try:
                return pd.read_csv(url)
            except Exception:
                return pd.DataFrame()

        diab_path = os.path.join(working_dir, 'Diabetes Prediction/diabetes.csv')
        df_diab  = pd.read_csv(diab_path) if os.path.exists(diab_path) else pd.DataFrame()
        heart_path = os.path.join(working_dir, 'Heart Disease Prediction/heart.csv')
        df_heart = pd.read_csv(heart_path) if os.path.exists(heart_path) else pd.DataFrame()
        pk_path  = os.path.join(working_dir, 'Parkinsons Disease Detection/parkinsons.csv')
        df_park  = pd.read_csv(pk_path) if os.path.exists(pk_path) else pd.DataFrame()
        bc       = load_breast_cancer()
        df_bc    = pd.DataFrame(bc.data, columns=bc.feature_names)
        df_bc['target']    = bc.target
        df_bc['diagnosis'] = df_bc['target'].map({1:'Benign', 0:'Malignant'})
        lv_path  = os.path.join(working_dir, 'indian_liver_patient.csv')
        df_liver = pd.read_csv(lv_path) if os.path.exists(lv_path) else pd.DataFrame()
        return df_diab, df_heart, df_park, df_bc, df_liver

    df_diab, df_heart, df_park, df_bc, df_liver = get_datasets()
    print(df_diab)

    kpis = [
        (len(df_diab)  if not df_diab.empty  else 768, "Diabetes Records"),
        (len(df_heart) if not df_heart.empty else 303,  "Heart Disease Records"),
        (len(df_park)  if not df_park.empty  else 195,  "Parkinson's Records"),
        (len(df_bc),                                    "Breast Cancer Records"),
        (len(df_liver) if not df_liver.empty else 583,  "Liver Disease Records"),
    ]
    cols = st.columns(5)
    for col, (val, lbl) in zip(cols, kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val:,}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🩸 Diabetes","❤️ Heart","🧠 Parkinson's","🎗️ Breast Cancer","🫀 Liver","🌍 Global Burden"])

    with tab1:
        if df_diab.empty:
            st.info("Add `diabetes.csv` to your project root to see local data. Using public fallback.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                vc = df_diab['Outcome'].value_counts().reset_index()
                vc.columns = ['Outcome','Count']
                vc['Label'] = vc['Outcome'].map({0:'Non-Diabetic',1:'Diabetic'})
                fig = px.pie(vc, values='Count', names='Label', hole=0.45,
                             color_discrete_sequence=['#0891b2','#e11d48'],
                             title='Diabetic vs Non-Diabetic')
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.histogram(df_diab, x='Glucose',
                                   color=df_diab['Outcome'].map({0:'Non-Diabetic',1:'Diabetic'}),
                                   barmode='overlay', nbins=35, opacity=0.75,
                                   color_discrete_map={'Non-Diabetic':'#0891b2','Diabetic':'#e11d48'},
                                   title='Glucose Distribution')
                fig.update_layout(legend_title='')
                st.plotly_chart(fig, width="stretch")
            with c3:
                fig = px.box(df_diab,
                             x=df_diab['Outcome'].map({0:'Non-Diabetic',1:'Diabetic'}),
                             y='BMI',
                             color=df_diab['Outcome'].map({0:'Non-Diabetic',1:'Diabetic'}),
                             color_discrete_map={'Non-Diabetic':'#0891b2','Diabetic':'#e11d48'},
                             title='BMI by Outcome')
                fig.update_layout(showlegend=False, xaxis_title='')
                st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(df_diab, x='Age', y='Glucose',
                                 color=df_diab['Outcome'].map({0:'Non-Diabetic',1:'Diabetic'}),
                                 color_discrete_map={'Non-Diabetic':'#0891b2','Diabetic':'#e11d48'},
                                 opacity=0.6, title='Age vs Glucose')
                fig.update_layout(legend_title='')
                st.plotly_chart(fig, width="stretch")
            with c2:
                corr = df_diab.corr(numeric_only=True)
                fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='Blues',
                                title='Feature Correlation Heatmap')
                st.plotly_chart(fig, width="stretch")

    with tab2:
        if df_heart.empty:
            st.info("Add `heart.csv` to your project root.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                vc = df_heart['target'].value_counts().reset_index()
                vc.columns = ['target','Count']
                vc['Label'] = vc['target'].map({0:'No Disease',1:'Heart Disease'})
                fig = px.bar(vc, x='Label', y='Count', color='Label',
                             color_discrete_map={'No Disease':'#0891b2','Heart Disease':'#e11d48'},
                             title='Disease Prevalence')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.scatter(df_heart, x='age', y='thalach',
                                 color=df_heart['target'].map({0:'No Disease',1:'Heart Disease'}),
                                 color_discrete_map={'No Disease':'#0891b2','Heart Disease':'#e11d48'},
                                 opacity=0.7, title='Age vs Max Heart Rate')
                fig.update_layout(legend_title='')
                st.plotly_chart(fig, width="stretch")
            with c3:
                cp_map = {0:'Typical',1:'Atypical',2:'Non-anginal',3:'Asymptomatic'}
                df_cp = df_heart.groupby(['cp','target']).size().reset_index(name='count')
                df_cp['cp_label'] = df_cp['cp'].map(cp_map)
                df_cp['disease']  = df_cp['target'].map({0:'No Disease',1:'Heart Disease'})
                fig = px.bar(df_cp, x='cp_label', y='count', color='disease', barmode='group',
                             color_discrete_map={'No Disease':'#0891b2','Heart Disease':'#e11d48'},
                             title='Chest Pain Type vs Disease')
                fig.update_layout(legend_title='', xaxis_title='')
                st.plotly_chart(fig, width="stretch")
            fig = px.violin(df_heart, x=df_heart['target'].map({0:'No Disease',1:'Heart Disease'}),
                            y='chol', color=df_heart['target'].map({0:'No Disease',1:'Heart Disease'}),
                            color_discrete_map={'No Disease':'#0891b2','Heart Disease':'#e11d48'},
                            box=True, title='Cholesterol Distribution by Outcome')
            fig.update_layout(showlegend=False, xaxis_title='')
            st.plotly_chart(fig, width="stretch")

    with tab3:
        if df_park.empty:
            st.info("Add `parkinsons.csv` (UCI dataset) to your project root.")
        else:
            status_col = 'status' if 'status' in df_park.columns else df_park.columns[-1]
            df_park['label'] = df_park[status_col].map({0:'Healthy',1:"Parkinson's"})
            c1, c2 = st.columns(2)
            with c1:
                vc = df_park['label'].value_counts().reset_index()
                vc.columns = ['label','count']
                fig = px.pie(vc, values='count', names='label', hole=0.4,
                             color_discrete_map={"Healthy":'#0891b2',"Parkinson's":'#e11d48'},
                             title="Healthy vs Parkinson's")
                st.plotly_chart(fig, width="stretch")
            with c2:
                if 'MDVP:Fo(Hz)' in df_park.columns and 'HNR' in df_park.columns:
                    fig = px.scatter(df_park, x='MDVP:Fo(Hz)', y='HNR', color='label',
                                     color_discrete_map={"Healthy":'#0891b2',"Parkinson's":'#e11d48'},
                                     opacity=0.7, title='Fundamental Frequency vs HNR')
                    fig.update_layout(legend_title='')
                    st.plotly_chart(fig, width="stretch")
            num_cols = [c for c in df_park.select_dtypes(include=np.number).columns if c != status_col][:6]
            if num_cols:
                fig = px.box(df_park, y=num_cols, color='label',
                             color_discrete_map={"Healthy":'#0891b2',"Parkinson's":'#e11d48'},
                             title='Voice Feature Distribution')
                fig.update_layout(legend_title='')
                st.plotly_chart(fig, width="stretch")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df_bc, names='diagnosis', hole=0.4,
                         color_discrete_map={'Benign':'#0891b2','Malignant':'#e11d48'},
                         title='Malignant vs Benign')
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.box(df_bc, x='diagnosis', y='mean radius', color='diagnosis',
                         color_discrete_map={'Benign':'#0891b2','Malignant':'#e11d48'},
                         title='Tumour Mean Radius')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df_bc, x='mean radius', y='mean texture', color='diagnosis',
                             color_discrete_map={'Benign':'#0891b2','Malignant':'#e11d48'},
                             opacity=0.65, title='Radius vs Texture')
            fig.update_layout(legend_title='')
            st.plotly_chart(fig, width="stretch")
        with c2:
            feats = ['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','target']
            corr = df_bc[feats].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                            title='Top Feature Correlations')
            st.plotly_chart(fig, width="stretch")

    with tab5:
        if df_liver.empty:
            st.info("Add `indian_liver_patient.csv` (Kaggle ILPD) to your project root.")
        else:
            target_col = 'Dataset' if 'Dataset' in df_liver.columns else df_liver.columns[-1]
            df_liver['label'] = df_liver[target_col].map({1:'Liver Disease',2:'No Disease'})
            c1, c2, c3 = st.columns(3)
            with c1:
                vc = df_liver['label'].value_counts().reset_index()
                vc.columns = ['label','count']
                fig = px.pie(vc, values='count', names='label', hole=0.4,
                             color_discrete_map={'Liver Disease':'#e11d48','No Disease':'#0891b2'},
                             title='Liver Disease Distribution')
                st.plotly_chart(fig, width="stretch")
            with c2:
                if 'Age' in df_liver.columns:
                    fig = px.histogram(df_liver, x='Age', color='label', barmode='overlay',
                                       color_discrete_map={'Liver Disease':'#e11d48','No Disease':'#0891b2'},
                                       opacity=0.75, title='Age Distribution')
                    fig.update_layout(legend_title='')
                    st.plotly_chart(fig, width="stretch")
            with c3:
                if 'Total_Bilirubin' in df_liver.columns:
                    fig = px.box(df_liver, x='label', y='Total_Bilirubin', color='label',
                                 color_discrete_map={'Liver Disease':'#e11d48','No Disease':'#0891b2'},
                                 title='Total Bilirubin')
                    fig.update_layout(showlegend=False, xaxis_title='')
                    st.plotly_chart(fig, width="stretch")

    with tab6:
        burden = pd.DataFrame({
            'Disease':          ['Diabetes','Heart Disease','Breast Cancer',"Parkinson's",'Kidney Disease','Liver Disease','Brain Tumour'],
            'Cases (Millions)': [537, 520, 2.3, 11.7, 850, 1500, 0.3],
            'Deaths (Millions)':[6.7, 17.9, 0.68, 0.33, 1.3, 2.0, 0.25],
            'Mortality %':      [11, 31, 15, 5, 18, 20, 77],
        })
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(burden, x='Disease', y='Cases (Millions)', color='Disease',
                         title='Estimated Global Cases (Millions)',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(burden, x='Cases (Millions)', y='Deaths (Millions)',
                             size='Mortality %', color='Disease', text='Disease',
                             title='Cases vs Deaths (bubble size = mortality %)',
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=burden['Disease'], y=burden['Cases (Millions)'],
                             name='Cases (M)', marker_color='#0891b2', opacity=0.8))
        fig.add_trace(go.Scatter(x=burden['Disease'], y=burden['Mortality %'],
                                 name='Mortality %', mode='lines+markers', yaxis='y2',
                                 marker=dict(color='#e11d48', size=9),
                                 line=dict(color='#e11d48', width=2.5)))
        fig.update_layout(
            title='Global Disease Cases vs Mortality Rate',
            yaxis=dict(title='Cases (Millions)'),
            yaxis2=dict(title='Mortality %', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.1),
        )
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  DIABETES
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Diabetes':
    st.markdown('<p class="main-title">Diabetes <span>Risk Prediction</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter patient vitals to assess diabetes risk.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        Pregnancies   = st.number_input('Pregnancies', 0, 20, 1)
        Glucose       = st.number_input('Glucose (mg/dL)', 0, 300, 110)
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', 0, 200, 72)
    with c2:
        SkinThickness = st.number_input('Skin Thickness (mm)', 0, 100, 23)
        Insulin       = st.number_input('Insulin (µU/mL)', 0, 900, 85)
        BMI           = st.number_input('BMI', 0.0, 70.0, 26.0, step=0.1)
    with c3:
        DPF = st.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.47, step=0.01)
        Age = st.number_input('Age', 1, 120, 31)
    if st.button('🔍 Predict Diabetes Risk', width="stretch"):
        if 'diabetes' in models:
            pred = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
            show_result(pred[0]==1, "High risk of Diabetes. Consult an endocrinologist.", "Low risk of Diabetes. Maintain a healthy lifestyle!")
        else:
            st.error("Model file `diabetes_model.sav` not found in `saved_models/`.")


# ═══════════════════════════════════════════════════════════════════════════════
#  HEART DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Heart Disease':
    st.markdown('<p class="main-title">Heart Disease <span>Prediction</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cardiovascular risk assessment using clinical parameters.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.number_input('Age', 1, 120, 52)
        sex      = st.selectbox('Sex', [('Male',1),('Female',0)], format_func=lambda x: x[0])
        cp       = st.selectbox('Chest Pain Type', [(0,'Typical Angina'),(1,'Atypical Angina'),(2,'Non-anginal'),(3,'Asymptomatic')], format_func=lambda x: x[1])
        trestbps = st.number_input('Resting BP (mm Hg)', 80, 220, 125)
        chol     = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 212)
    with c2:
        fbs     = st.selectbox('Fasting Blood Sugar > 120', [('No',0),('Yes',1)], format_func=lambda x: x[0])
        restecg = st.selectbox('Resting ECG', [(0,'Normal'),(1,'ST-T Abnormality'),(2,'LV Hypertrophy')], format_func=lambda x: x[1])
        thalach = st.number_input('Max Heart Rate', 60, 220, 168)
        exang   = st.selectbox('Exercise Induced Angina', [('No',0),('Yes',1)], format_func=lambda x: x[0])
    with c3:
        oldpeak = st.number_input('ST Depression', 0.0, 7.0, 1.0, step=0.1)
        slope   = st.selectbox('ST Slope', [(0,'Upsloping'),(1,'Flat'),(2,'Downsloping')], format_func=lambda x: x[1])
        ca      = st.number_input('Major Vessels (0–4)', 0, 4, 2)
        thal    = st.selectbox('Thal', [(0,'Normal'),(1,'Fixed Defect'),(2,'Reversible Defect')], format_func=lambda x: x[1])
    if st.button('🔍 Predict Heart Disease Risk', width="stretch"):
        if 'heart' in models:
            inp  = [age, sex[1], cp[0], trestbps, chol, fbs[1], restecg[0], thalach, exang[1], oldpeak, slope[0], ca, thal[0]]
            pred = models['heart'].predict([inp])
            show_result(pred[0]==1, "Signs of Heart Disease detected. Consult a cardiologist.", "No significant heart disease risk detected.")
        else:
            st.error("Model file `heart_disease_model.sav` not found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PARKINSON'S
# ═══════════════════════════════════════════════════════════════════════════════
if selected == "Parkinson's":
    st.markdown("<p class='main-title'>Parkinson's <span>Disease Prediction</span></p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Voice biomarker analysis for early Parkinson's detection.</p>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo             = st.number_input('MDVP:Fo(Hz)', value=88.33300, format='%.5f')
        fhi            = st.number_input('MDVP:Fhi(Hz)', value=112.24000, format='%.5f')
        flo            = st.number_input('MDVP:Flo(Hz)', value=84.07200, format='%.5f')
        Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.00505, format='%.5f')
        Jitter_Abs     = st.number_input('MDVP:Jitter(Abs)', value=0.00006, format='%.5f')

    with col2:
        RAP        = st.number_input('MDVP:RAP', value=0.00254, format='%.5f')
        PPQ        = st.number_input('MDVP:PPQ', value=0.00330, format='%.5f')
        DDP        = st.number_input('Jitter:DDP', value=0.00763, format='%.5f')
        Shimmer    = st.number_input('MDVP:Shimmer', value=0.02143, format='%.5f')
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=0.19700, format='%.5f')

    with col3:
        APQ3 = st.number_input('Shimmer:APQ3', value=0.01079, format='%.5f')
        APQ5 = st.number_input('Shimmer:APQ5', value=0.01342, format='%.5f')
        APQ  = st.number_input('MDVP:APQ', value=0.01892, format='%.5f')
        DDA  = st.number_input('Shimmer:DDA', value=0.03237, format='%.5f')

    with col4:
        NHR  = st.number_input('NHR', value=0.01166, format='%.5f')
        HNR  = st.number_input('HNR', value=21.11800, format='%.5f')
        RPDE = st.number_input('RPDE', value=0.61114, format='%.5f')
        DFA  = st.number_input('DFA', value=0.77616, format='%.5f')

    with col5:
        spread1 = st.number_input('spread1', value=-5.24977, format='%.5f')
        spread2 = st.number_input('spread2', value=0.39100, format='%.5f')
        D2  = st.number_input('D2', value=2.40731, format='%.5f')
        PPE = st.number_input('PPE', value=0.249740, format='%.5f')

    user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
    if st.button("🔍 Predict Parkinson's Risk", width="stretch"):
        if 'parkinsons' in models:
            pred = models['parkinsons'].predict([user_input])
            show_result(pred[0]==1, "Parkinson's indicators detected. Consult a neurologist.", "No significant Parkinson's indicators found.")
        else:
            st.error("Model file `parkinsons_model.sav` not found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAIN TUMOUR
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Brain Tumour':
    st.markdown('<p class="main-title">Brain Tumour <span>Detection</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an MRI scan — VGG16 deep learning classifies the tumour type.</p>', unsafe_allow_html=True)
    CLASS_LABELS = ['pituitary','glioma','notumor','meningioma']
    IMAGE_SIZE   = 128
    TUMOUR_INFO  = {
        'pituitary':  ('Pituitary Tumour','🟠','Forms in the pituitary gland at the base of the brain.'),
        'glioma':     ('Glioma','🔴','Starts in glial cells of the brain or spinal cord.'),
        'meningioma': ('Meningioma','🟡','Arises from membranes surrounding the brain.'),
        'notumor':    ('No Tumour Detected','🟢','No signs of a brain tumour in this MRI scan.'),
    }
    uploaded = st.file_uploader('Upload Brain MRI Image', type=['jpg','jpeg','png'])
    if uploaded:
        from PIL import Image as PILImage
        image = PILImage.open(uploaded).convert('RGB')
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption='Uploaded MRI Scan', width="stretch")
        with c2:
            if st.button('🔍 Detect Tumour', width="stretch"):
                with st.spinner('Analysing with VGG16...'):
                    bm = load_brain_model()
                    if bm:
                        arr  = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
                        arr  = np.expand_dims(arr, 0)
                        pred = bm.predict(arr)
                        idx  = int(np.argmax(pred, axis=1)[0])
                        conf = float(np.max(pred)) * 100
                        lbl  = CLASS_LABELS[idx]
                        title, icon, desc = TUMOUR_INFO[lbl]
                        (st.error if lbl != 'notumor' else st.success)(f'{icon} **{title}**')
                        st.metric('Confidence', f'{conf:.2f}%')
                        st.info(desc)
                        st.subheader('Class Probabilities')
                        for i, cl in enumerate(CLASS_LABELS):
                            p = float(pred[0][i]) * 100
                            st.progress(int(p), text=f'{TUMOUR_INFO[cl][0]}: {p:.1f}%')
                    else:
                        st.warning('Add `brain_tumor_model.keras` to `saved_models/`.')
    else:
        st.info('👆 Upload a brain MRI scan (JPG/PNG) to begin.')
        cols = st.columns(4)
        for col, (k,(t,ic,d)) in zip(cols, TUMOUR_INFO.items()):
            with col:
                st.markdown(f'### {ic}')
                st.markdown(f'**{t}**')
                st.caption(d)


# ═══════════════════════════════════════════════════════════════════════════════
#  BREAST CANCER
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Breast Cancer':
    st.markdown('<p class="main-title">Breast Cancer <span>Detection</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Classify tumour as Malignant or Benign using cell nucleus measurements from FNA biopsy images.</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📊 Features computed from digitized FNA images. Defaults shown are dataset mean values.</div>', unsafe_allow_html=True)

    def feat_col(col, items):
        out = {}
        with col:
            for lbl, key, default in items:
                out[key] = st.number_input(lbl, value=float(default), format='%.4f')
        return out

    c1, c2, c3 = st.columns(3)
    f1 = feat_col(c1,[('Mean Radius','r1',14.127),('Mean Texture','t1',19.290),
        ('Mean Perimeter','p1',91.969),('Mean Area','a1',654.889),('Mean Smoothness','sm1',0.0964),
        ('Mean Compactness','co1',0.1043),('Mean Concavity','cv1',0.0888),
        ('Mean Concave Pts','cp1',0.0489),('Mean Symmetry','sy1',0.1812),('Mean Fractal','fr1',0.0628)])
    f2 = feat_col(c2,[('SE Radius','r2',0.4052),('SE Texture','t2',1.2169),
        ('SE Perimeter','p2',2.8661),('SE Area','a2',40.337),('SE Smoothness','sm2',0.0070),
        ('SE Compactness','co2',0.0255),('SE Concavity','cv2',0.0319),
        ('SE Concave Pts','cp2',0.0118),('SE Symmetry','sy2',0.0205),('SE Fractal','fr2',0.0038)])
    f3 = feat_col(c3,[('Worst Radius','r3',16.269),('Worst Texture','t3',25.677),
        ('Worst Perimeter','p3',107.26),('Worst Area','a3',880.58),('Worst Smoothness','sm3',0.1324),
        ('Worst Compactness','co3',0.2543),('Worst Concavity','cv3',0.2722),
        ('Worst Concave Pts','cp3',0.1146),('Worst Symmetry','sy3',0.2901),('Worst Fractal','fr3',0.0839)])

    if st.button('🔍 Predict Breast Cancer', width="stretch"):
        if 'breast_cancer' in models:
            features = list(f1.values()) + list(f2.values()) + list(f3.values())
            pred = models['breast_cancer'].predict([features])
            show_result(pred[0]==0,
                "Tumour classified as <strong>Malignant</strong>. Immediate oncology consultation recommended.",
                "Tumour classified as <strong>Benign</strong>. Continue regular screening and monitoring.")
        else:
            st.error("Model file `breast_cancer_model.sav` not found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  KIDNEY DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Kidney Disease':
    st.markdown('<p class="main-title">Kidney Disease <span>Prediction</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict Chronic Kidney Disease (CKD) from blood and urine test parameters.</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age  = st.number_input('Age', 1, 100, 72)
        bp   = st.number_input('Blood Pressure (mm Hg)', 50, 180, 60)
        sg   = st.selectbox('Specific Gravity', [1.005,1.010,1.015,1.020,1.025])
        al   = st.selectbox('Albumin (0–5)', [0,1,2,3,4,5])
        su   = st.selectbox('Sugar (0–5)', [0,1,2,3,4,5])
        rbc  = st.selectbox('Red Blood Cells', [('Normal',0),('Abnormal',1)], format_func=lambda x: x[0])
    with c2:
        pc   = st.selectbox('Pus Cell', [('Normal',0),('Abnormal',1)], format_func=lambda x: x[0])
        pcc  = st.selectbox('Pus Cell Clumps', [('Not Present',0),('Present',1)], format_func=lambda x: x[0])
        ba   = st.selectbox('Bacteria', [('Not Present',0),('Present',1)], format_func=lambda x: x[0])
        bgr  = st.number_input('Blood Glucose Random (mg/dl)', 50, 500, 109)
        bu   = st.number_input('Blood Urea (mg/dl)', 1, 400, 26)
        sc   = st.number_input('Serum Creatinine (mg/dl)', 0.0, 20.0, 0.90, step=0.1)
    with c3:
        sod  = st.number_input('Sodium (mEq/L)', 100, 180, 150)
        pot  = st.number_input('Potassium (mEq/L)', 2.0, 10.0, 4.90, step=0.1)
        hemo = st.number_input('Haemoglobin (g/dL)', 3.0, 18.0, 15.0, step=0.1)
        pcv  = st.number_input('Packed Cell Volume', 10, 60, 52)
        wc   = st.number_input('WBC Count (cells/cumm)', 2000, 26000, 10500)
        rc   = st.number_input('RBC Count (mil/cmm)', 1.0, 8.0, 5.50, step=0.1)
    with c4:
        htn   = st.selectbox('Hypertension', [('No',0),('Yes',1)], format_func=lambda x: x[0])
        dm    = st.selectbox('Diabetes Mellitus', [('No',0),('Yes',1)], format_func=lambda x: x[0])
        cad   = st.selectbox('Coronary Artery Disease', [('No',0),('Yes',1)], format_func=lambda x: x[0])
        appet = st.selectbox('Appetite', [('Good',1),('Poor',0)], format_func=lambda x: x[0])
        pe    = st.selectbox('Pedal Edema', [('No',0),('Yes',1)], format_func=lambda x: x[0])
        ane   = st.selectbox('Anaemia', [('No',0),('Yes',1)], format_func=lambda x: x[0])
    if st.button('🔍 Predict Kidney Disease', width="stretch"):
        if 'kidney' in models:
            features = [age,bp,sg,al,su,rbc[1],pc[1],pcc[1],ba[1],bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn[1],dm[1],cad[1],appet[1],pe[1],ane[1]]
            pred = models['kidney'].predict([features])
            show_result(pred[0]==1,
                "Chronic Kidney Disease detected. Consult a nephrologist immediately.",
                "No CKD detected. Stay hydrated and have regular checkups.")
        else:
            st.error("Model file `kidney_disease_model.sav` not found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVER DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Liver Disease':
    st.markdown('<p class="main-title">Liver Disease <span>Prediction</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict liver disease using standard liver function test (LFT) values.</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📊 Based on the Indian Liver Patient Dataset (ILPD) — 583 patient records.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.number_input('Age', 1, 100, 45)
        gender   = st.selectbox('Gender', [('Male',1),('Female',0)], format_func=lambda x: x[0])
        tot_bili = st.number_input('Total Bilirubin', 0.0, 75.0, 0.9, step=0.1)
        dir_bili = st.number_input('Direct Bilirubin', 0.0, 20.0, 0.2, step=0.1)
    with c2:
        alk_phos  = st.number_input('Alkaline Phosphotase', 50, 2200, 290)
        alamine   = st.number_input('Alamine Aminotransferase (ALT)', 1, 2000, 25)
        aspartate = st.number_input('Aspartate Aminotransferase (AST)', 1, 5000, 35)
    with c3:
        tot_prot = st.number_input('Total Proteins (g/dL)', 0.0, 10.0, 6.8, step=0.1)
        albumin  = st.number_input('Albumin (g/dL)', 0.0, 6.0, 3.3, step=0.1)
        ag_ratio = st.number_input('A/G Ratio', 0.0, 3.0, 1.0, step=0.01)
    if st.button('🔍 Predict Liver Disease', width="stretch"):
        if 'liver' in models:
            features = [age, gender[1], tot_bili, dir_bili, alk_phos, alamine, aspartate, tot_prot, albumin, ag_ratio]
            pred = models['liver'].predict([features])
            show_result(pred[0]==1,
                "Liver Disease indicators detected. Consult a hepatologist promptly.",
                "No significant Liver Disease indicators found. Maintain a healthy diet.")
        else:
            st.error("Model file `liver_disease_model.sav` not found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  AI HEALTH CHATBOT  — Hugging Face Mixtral (free, no API key required)
# ═══════════════════════════════════════════════════════════════════════════════
if selected == 'Health Chatbot':
    st.markdown('<p class="main-title">🤖 AI Health <span>Chatbot</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Describe your symptoms — powered by Dolphin Mistral 24B Venice Edition on Hugging Face (free, no API key needed).</p>', unsafe_allow_html=True)
    st.warning("⚠️ **Medical Disclaimer:** General health information only. Always consult a qualified healthcare professional for diagnosis and treatment.")

    SYSTEM_CONTEXT = (
        "You are MedAI, an expert AI health assistant embedded in a Multiple Disease Prediction System "
        "covering Diabetes, Heart Disease, Parkinson's Disease, Brain Tumours, Breast Cancer, Kidney Disease, and Liver Disease. "
        "Give clear, empathetic, structured responses using bullet points and emojis. "
        "Relate symptoms to the disease categories listed above where relevant. "
        "Always end with a reminder to see a real doctor. Never definitively diagnose — say 'may indicate' or 'could be related to'. "
        "Keep responses to 150–250 words max."
    )

    def query_hf(history):
        # Build conversation messages list
        messages = []
        for m in history[1:]:
            if m['role'] == 'user':
                messages.append({"role": "user", "content": m['content']})
            else:
                messages.append({"role": "assistant", "content": m['content']})

        try:
            # Initialize client
            hf_token = st.secrets.get("HF_TOKEN", "")
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )

            # Call completion
            completion = client.chat.completions.create(
                model="dphn/Dolphin-Mistral-24B-Venice-Edition:featherless-ai",
                messages=messages,
                max_tokens=350,
                temperature=0.7,
                top_p=0.9,
            )

            # Extract response text
            text = completion.choices[0].message.content.strip()
            return text if text else "I couldn't generate a response. Please try again."

        except Exception as e:
            print(e)
            return (
                "⚠️ Could not reach the AI model. General health tips:\n\n"
                "🩺 See a doctor for any persistent symptoms\n"
                "💧 Stay well hydrated — aim for 8 glasses of water daily\n"
                "🏃 30 mins of moderate exercise 5× per week reduces most chronic disease risks\n"
                "😴 7–9 hours of sleep supports immune function and metabolism"
            )

    # ── Session state ──────────────────────────────────────────────────────────
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{'role':'assistant','content':(
            "👋 Hello! I'm **MedAI**, your AI Health Assistant.\n\n"
            "I cover: Diabetes · Heart Disease · Parkinson's · Brain Tumours · Breast Cancer · Kidney Disease · Liver Disease\n\n"
            "Describe your symptoms or ask a health question to get started! 😊"
        )}]

    # Quick prompts
    st.markdown("**Quick questions:**")
    qp_cols = st.columns(4)
    quick_prompts = [
        "Early signs of diabetes?",
        "How to lower heart risks?",
        "Symptoms of kidney disease?",
        "Liver disease warning signs?",
    ]
    for col, prompt in zip(qp_cols, quick_prompts):
        if col.button(prompt, width="stretch"):
            st.session_state['_quick'] = prompt
    st.markdown("---")

    # Display history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user">👤 <strong>You</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            html_content = markdown.markdown(msg['content'])
            st.markdown(f'<div class="chat-bot">🤖 <strong>MedAI</strong><br>{html_content}</div>', unsafe_allow_html=True)


    # Input
    user_input = st.chat_input("Describe your symptoms or ask a health question…")
    if '_quick' in st.session_state:
        user_input = st.session_state.pop('_quick')

    if user_input:
        st.session_state.chat_history.append({'role':'user','content':user_input})
        with st.spinner('MedAI is thinking…'):
            reply = query_hf(st.session_state.chat_history)
        st.session_state.chat_history.append({'role':'assistant','content':reply})
        st.rerun()

    if len(st.session_state.chat_history) > 1:
        if st.button('🗑️ Clear Chat'):
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.rerun()

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem;color:#94a3b8;'>
💡 <strong>Tip:</strong> Add a free <a href='https://huggingface.co/settings/tokens' target='_blank'>Hugging Face token</a>
to Streamlit Cloud Secrets as <code>HF_TOKEN</code> for faster responses and higher rate limits.
Model: <strong>Dolphin-Mistral-24B-Venice-Edition</strong>
</div>
""", unsafe_allow_html=True)