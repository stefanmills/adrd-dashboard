"""
ADRD Cognitive Status Dashboard — UTRGV
Streamlit version for deployment and sharing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ADRD Cognitive Status Dashboard — UTRGV",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UTRGV Colors
UTRGV_GREEN = "#00573F"
UTRGV_ORANGE = "#F15A22"
UTRGV_DARK = "#2D2D2D"

st.markdown(f"""
<style>
    .stApp {{background-color: #F5F5F0;}}
    header[data-testid="stHeader"] {{background-color: {UTRGV_GREEN};}}
    .stTabs [data-baseweb="tab-list"] {{gap: 2px;}}
    .stTabs [data-baseweb="tab"] {{
        background-color: white; border-radius: 4px 4px 0 0;
        padding: 10px 20px; color: {UTRGV_DARK};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {UTRGV_GREEN}; color: white;
    }}
    div[data-testid="metric-container"] {{
        background-color: white; border: 1px solid #E0E0E0;
        border-left: 3px solid {UTRGV_GREEN}; padding: 12px; border-radius: 8px;
    }}
    .block-container {{padding-top: 1rem;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv.gz", compression="gzip")
    sentinel = [888, 888.8, 999, 995, -4, -4.4, 88]
    for col in ['NACCAGE','NACCBMI','BPSYS','BPDIAS','CDRSUM','ANIMALS','TRAILA','NACCGDS','EDUC','HRATE']:
        if col in df.columns:
            df[col] = df[col].replace(sentinel, np.nan)
    # Labels
    df['Sex'] = df['SEX'].map({1:'Male',2:'Female'}).fillna('Unknown')
    df['Race'] = df['RACE'].map({1:'White',2:'Black/AA',3:'Am Indian',4:'Pacific Isl.',5:'Asian',50:'Other'}).fillna('Unknown')
    df['CogStatus'] = df['NORMCOG'].map({1:'Normal',0:'Impaired'}).fillna('Unknown')
    df['MaritalStatus'] = df['MARISTAT'].map({1:'Married',2:'Widowed',3:'Divorced',4:'Separated',5:'Never married',6:'Other'}).fillna('Unknown')
    df['APOE_e4'] = df['NACCNE4S'].map({0:'0 copies',1:'1 copy',2:'2 copies'}).fillna('Unknown')
    return df

df = load_data()

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    import os, gzip, pickle
    def load_pkl(name):
        # Try compressed first, then uncompressed
        if os.path.exists(name + '.gz'):
            with gzip.open(name + '.gz', 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(name):
            return joblib.load(name)
        else:
            raise FileNotFoundError(f'{name} not found')
    try:
        model = load_pkl('normcog_xgb_lasso_model.pkl')
        features = load_pkl('normcog_lasso_features.pkl')
        imp = load_pkl('normcog_imputer.pkl')
        return model, features, imp, True
    except Exception as e:
        st.warning(f'Model not loaded: {e}')
        return None, [], None, False

xgb_model, lasso_features, imputer, model_loaded = load_model()

# Label maps
FEAT_LABELS = {
    'NACCAGE':'Age','SEX':'Sex','EDUC':'Education (yrs)','NACCNE4S':'APOE e4',
    'CDRSUM':'CDR Sum','CDRGLOB':'CDR Global','MEMORY':'CDR Memory',
    'ORIENT':'CDR Orientation','JUDGMENT':'CDR Judgment','ANIMALS':'Animal Fluency',
    'TRAILA':'Trail Making A','NACCGDS':'Depression (GDS)',
    'DIABETES':'Diabetes','HYPERTEN':'Hypertension','NACCTBI':'Head Injury (TBI)',
    'DEP2YRS':'Depression (2yr)','CANCER':'Cancer','NACCBMI':'BMI',
    'BPSYS':'Systolic BP','BPDIAS':'Diastolic BP','HRATE':'Heart Rate',
    'TOBAC100':'Tobacco Use','ALCOHOL':'Alcohol Abuse','MARISTAT':'Marital Status',
    'CBSTROKE':'Stroke Hx','CBTIA':'TIA Hx','CVHATT':'Heart Attack Hx',
    'CVAFIB':'Atrial Fibrillation','CVBYPASS':'Bypass Surgery',
    'CVCHF':'Congestive Heart Failure','CVPACE':'Pacemaker',
    'CVOTHR':'Other CV Condition','HYPERTEN':'Hypertension',
    'MYOINF':'Myocardial Infarction','CONGHRT':'Congestive Heart',
    'AFIBRILL':'Atrial Fibrillation (Med Hx)','ANGINA':'Angina',
    'HVALVE':'Heart Valve Disease','PACEMAKE':'Pacemaker (Med Hx)',
    'ANGIOCP':'Angioplasty/Endarterectomy','ANGIOPCI':'Angioplasty/Stent',
    'INCONTU':'Urinary Incontinence','INCONTF':'Fecal Incontinence',
    'URINEINC':'Urinary Incontinence (Med Hx)','BOWLINC':'Bowel Incontinence',
    'B12DEF':'B12 Deficiency','THYROID':'Thyroid Disease','VB12DEF':'Vitamin B12 Def',
    'SEIZURES':'Seizures','PD':'Parkinson Disease',
    'NACCADMD':'AD Medication','NACCNCRD':'Co-Diagnoses',
    'INDEPEND':'Independence Level','RACE':'Race','HISPANIC':'Hispanic',
}
VAL_LABELS = {
    'SEX':{1:'Male',2:'Female'},'NACCNE4S':{0:'0 copies',1:'1 copy',2:'2 copies'},
    'MARISTAT':{1:'Married',2:'Widowed',3:'Divorced',4:'Separated',5:'Never married',6:'Other'},
}
YN_FEATS = ['DIABETES','HYPERTEN','NACCTBI','DEP2YRS','TOBAC100','ALCOHOL','CANCER',
            'CBSTROKE','CBTIA','CVHATT','CVAFIB','CVBYPASS','CVCHF','CVPACE','CVOTHR',
            'MYOINF','CONGHRT','AFIBRILL','ANGINA','HVALVE','PACEMAKE','ANGIOCP','ANGIOPCI',
            'INCONTU','INCONTF','URINEINC','BOWLINC','B12DEF','THYROID','VB12DEF',
            'SEIZURES','PD','HISPANIC']
for f in YN_FEATS:
    VAL_LABELS[f] = {0:'No',1:'Yes'}

def get_label(f):
    if f in FEAT_LABELS: return FEAT_LABELS[f]
    if f.endswith('__missing'): return f'{FEAT_LABELS.get(f.replace("__missing",""), f.replace("__missing",""))} (data avail.)'
    return f

def get_val_label(f, v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
    base = f.replace('__missing','') if f.endswith('__missing') else f
    if base in VAL_LABELS: return VAL_LABELS[base].get(int(v), str(v))
    if isinstance(v, float): return f'{v:.1f}' if v != int(v) else str(int(v))
    return str(v)

# ============================================================
# HEADER
# ============================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <div style="background-color:{UTRGV_GREEN};padding:16px 24px;border-radius:8px;margin-bottom:16px;">
        <h1 style="color:white;margin:0;font-size:24px;">🧠 ADRD Cognitive Status Dashboard</h1>
        <p style="color:#A8D5BA;margin:0;font-size:13px;">NACC UDS — University of Texas Rio Grande Valley</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown(f"### 🔍 Filters")
age_range = st.sidebar.slider("Age Range", 18, 110, (18, 110))
sex_filter = st.sidebar.multiselect("Sex", ['Male','Female'], default=['Male','Female'])
race_filter = st.sidebar.multiselect("Race", df['Race'].unique().tolist(), default=df['Race'].unique().tolist())
normcog_filter = st.sidebar.selectbox("Cognitive Status", ['All','Normal','Impaired'])
diabetes_filter = st.sidebar.selectbox("Diabetes", ['All','Yes','No'])
htn_filter = st.sidebar.selectbox("Hypertension", ['All','Yes','No'])
dep_filter = st.sidebar.selectbox("Depression (2yr)", ['All','Yes','No'])
tbi_filter = st.sidebar.selectbox("TBI", ['All','Yes','No'])
cancer_filter = st.sidebar.selectbox("Cancer", ['All','Yes','No'])
alcohol_filter = st.sidebar.selectbox("Alcohol Abuse", ['All','Yes','No'])
marital_filter = st.sidebar.selectbox("Marital Status", ['All'] + sorted(df['MaritalStatus'].unique().tolist()))

# Apply filters
dff = df.copy()
dff = dff[(dff['NACCAGE'] >= age_range[0]) & (dff['NACCAGE'] <= age_range[1])]
if sex_filter: dff = dff[dff['Sex'].isin(sex_filter)]
if race_filter: dff = dff[dff['Race'].isin(race_filter)]
if normcog_filter != 'All': dff = dff[dff['CogStatus'] == normcog_filter]
yn_map = {'Yes': 1, 'No': 0}
if diabetes_filter != 'All': dff = dff[dff['DIABETES'] == yn_map[diabetes_filter]]
if htn_filter != 'All': dff = dff[dff['HYPERTEN'] == yn_map[htn_filter]]
if dep_filter != 'All': dff = dff[dff['DEP2YRS'] == yn_map[dep_filter]]
if tbi_filter != 'All': dff = dff[dff['NACCTBI'] == yn_map[tbi_filter]]
if cancer_filter != 'All' and 'CANCER' in dff.columns: dff = dff[dff['CANCER'] == yn_map[cancer_filter]]
if alcohol_filter != 'All': dff = dff[dff['ALCOHOL'] == yn_map[alcohol_filter]]
if marital_filter != 'All': dff = dff[dff['MaritalStatus'] == marital_filter]

st.sidebar.markdown(f"**Showing {len(dff):,}** of {len(df):,} records")

# ============================================================
# TABS
# ============================================================
tab_viz, tab_table, tab_summary, tab_predict = st.tabs(["📊 Visualizations", "📋 Data Table", "📈 Summary", "🔮 Prediction"])

# ============================================================
# TAB 1: VISUALIZATIONS
# ============================================================
with tab_viz:
    # Stat cards
    n = len(dff)
    nc_n = (dff['NORMCOG']==1).sum() if 'NORMCOG' in dff.columns else 0
    imp_n = (dff['NORMCOG']==0).sum() if 'NORMCOG' in dff.columns else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total", f"{n:,}")
    c2.metric("Normal", f"{nc_n:,}", f"{nc_n/n*100:.1f}%" if n>0 else "")
    c3.metric("Impaired", f"{imp_n:,}", f"{imp_n/n*100:.1f}%" if n>0 else "")
    c4.metric("Mean Age", f"{dff['NACCAGE'].mean():.1f}" if n>0 else "—")
    c5.metric("Mean BMI", f"{dff['NACCBMI'].mean():.1f}" if n>0 else "—")
    c6.metric("Mean CDR", f"{dff['CDRSUM'].mean():.1f}" if n>0 else "—")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        valid = dff[dff['CogStatus']!='Unknown']
        fig1 = px.histogram(valid, x='NACCAGE', color='CogStatus', nbins=40, barmode='overlay', opacity=0.7,
            color_discrete_map={'Normal':UTRGV_GREEN,'Impaired':UTRGV_ORANGE},
            title='Age Distribution by Cognitive Status')
        fig1.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        apoe = dff[dff['APOE_e4']!='Unknown'].groupby('APOE_e4').agg(
            total=('NORMCOG','count'),impaired=('NORMCOG',lambda x:(x==0).sum())).reset_index()
        apoe['Rate'] = (apoe['impaired']/apoe['total']*100).round(1)
        fig2 = px.bar(apoe, x='APOE_e4', y='Rate', text='Rate', title='APOE e4 vs Impairment Rate (%)',
            color='Rate', color_continuous_scale=[UTRGV_GREEN,'#D32F2F'])
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380,coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        coms = ['DIABETES','HYPERTEN','DEP2YRS','NACCTBI','CANCER']
        clabs = ['Diabetes','Hypertension','Depression','TBI','Cancer']
        cd = []
        for col_name, lab in zip(coms, clabs):
            if col_name in dff.columns:
                for nv, nl in [(1,'Normal'),(0,'Impaired')]:
                    sub = dff[dff['NORMCOG']==nv]
                    rate = sub[col_name].mean()*100 if len(sub)>0 else 0
                    cd.append({'Comorbidity':lab,'Group':nl,'Prevalence (%)':rate})
        fig3 = px.bar(pd.DataFrame(cd), x='Comorbidity', y='Prevalence (%)', color='Group', barmode='group',
            color_discrete_map={'Normal':UTRGV_GREEN,'Impaired':UTRGV_ORANGE},
            title='Comorbidity Prevalence by Cognitive Status')
        fig3.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        # Cardiovascular conditions by cognitive status
        cv_vars = ['CVHATT','CBSTROKE','CVAFIB','CVCHF','HYPERTEN']
        cv_labs = ['Heart Attack','Stroke','A-Fib','CHF','Hypertension']
        cv_data = []
        for col_name, lab in zip(cv_vars, cv_labs):
            if col_name in dff.columns:
                for nv, nl in [(1,'Normal'),(0,'Impaired')]:
                    sub = dff[dff['NORMCOG']==nv]
                    rate = sub[col_name].mean()*100 if len(sub)>0 else 0
                    cv_data.append({'Condition':lab,'Group':nl,'Prevalence (%)':rate})
        fig4 = px.bar(pd.DataFrame(cv_data), x='Condition', y='Prevalence (%)', color='Group', barmode='group',
            color_discrete_map={'Normal':UTRGV_GREEN,'Impaired':UTRGV_ORANGE},
            title='Cardiovascular Conditions by Cognitive Status')
        fig4.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380)
        st.plotly_chart(fig4, use_container_width=True)

    # Charts row 3
    col5, col6 = st.columns(2)
    with col5:
        sc = dff[dff['CogStatus']!='Unknown'].dropna(subset=['NACCAGE','CDRSUM'])
        fig5 = px.scatter(sc, x='NACCAGE', y='CDRSUM', color='CogStatus', opacity=0.4,
            color_discrete_map={'Normal':UTRGV_GREEN,'Impaired':UTRGV_ORANGE},
            title='Age vs CDR Sum by Cognitive Status')
        fig5.update_traces(marker=dict(size=4))
        fig5.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380)
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        sex_cog = dff[dff['CogStatus']!='Unknown'].groupby(['Sex','CogStatus']).size().reset_index(name='Count')
        fig6 = px.bar(sex_cog, x='Sex', y='Count', color='CogStatus', barmode='group',
            color_discrete_map={'Normal':UTRGV_GREEN,'Impaired':UTRGV_ORANGE},
            title='Cognitive Status by Sex')
        fig6.update_layout(plot_bgcolor='white',paper_bgcolor='white',height=380)
        st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# TAB 2: DATA TABLE
# ============================================================
with tab_table:
    st.subheader(f"Filtered Data — {len(dff):,} records")
    display_cols = ['NACCAGE','Sex','Race','MaritalStatus','EDUC','CogStatus','CDRSUM',
                    'NACCBMI','BPSYS','APOE_e4']
    col_rename = {'NACCAGE':'Age','MaritalStatus':'Marital Status','EDUC':'Education',
                  'CogStatus':'Cognitive Status','CDRSUM':'CDR Sum','NACCBMI':'BMI',
                  'BPSYS':'Systolic BP','APOE_e4':'APOE e4'}
    avail = [c for c in display_cols if c in dff.columns]
    tdf = dff[avail].head(500).rename(columns=col_rename)
    st.dataframe(tdf, use_container_width=True, height=600)

# ============================================================
# TAB 3: SUMMARY
# ============================================================
with tab_summary:
    st.subheader("Data Summary")
    n = len(dff)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"**Demographics**")
        st.write(f"Total records: **{n:,}**")
        st.write(f"Age — Mean: {dff['NACCAGE'].mean():.1f}, Range: {dff['NACCAGE'].min():.0f}–{dff['NACCAGE'].max():.0f}")
        st.write(f"Education — Mean: {dff['EDUC'].mean():.1f} years")
        sex_c = dff['Sex'].value_counts()
        st.write(f"Sex — " + ", ".join([f"{k}: {v:,} ({v/n*100:.1f}%)" for k,v in sex_c.items()]))

    with col_s2:
        st.markdown(f"**Cognitive Status**")
        nc_n = (dff['NORMCOG']==1).sum()
        st.write(f"Normal: **{nc_n:,}** ({nc_n/n*100:.1f}%)" if n>0 else "Normal: —")
        st.write(f"Impaired: **{n-nc_n:,}** ({(n-nc_n)/n*100:.1f}%)" if n>0 else "Impaired: —")

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.markdown(f"**Clinical Measures**")
        st.write(f"CDR Sum — Mean: {dff['CDRSUM'].mean():.2f}")
        st.write(f"BMI — Mean: {dff['NACCBMI'].mean():.1f}")
        st.write(f"Systolic BP — Mean: {dff['BPSYS'].mean():.1f}" if dff['BPSYS'].notna().any() else "BP: N/A")

    with col_s4:
        st.markdown(f"**Comorbidities**")
        for col_name, label in [('DIABETES','Diabetes'),('HYPERTEN','Hypertension'),('DEP2YRS','Depression'),
                         ('NACCTBI','TBI'),('CANCER','Cancer'),('ALCOHOL','Alcohol Abuse')]:
            if col_name in dff.columns:
                ct = (dff[col_name]==1).sum()
                st.write(f"{label}: {ct:,} ({(dff[col_name]==1).mean()*100:.1f}%)")

# ============================================================
# TAB 4: PREDICTION
# ============================================================
with tab_predict:
    if not model_loaded:
        st.error("Model files not found. Run the NORMCOG notebook first to generate .pkl files.")
    else:
        st.subheader("Cognitive Status Prediction")
        st.caption("Enter patient parameters below. The model uses 399 LASSO-selected features — you input the key ones, the rest auto-fill with population medians.")

        col_left, col_right = st.columns([2, 3])

        with col_left:
            # --- DEMOGRAPHICS ---
            st.markdown(f"#### 👤 Demographics")
            p_age = st.slider("Age", 50, 100, 75)
            p_sex = st.selectbox("Sex", [('Female',2),('Male',1)], format_func=lambda x: x[0])
            p_educ = st.slider("Education (years)", 0, 30, 16)
            p_race = st.selectbox("Race", [(v,k) for k,v in {1:'White',2:'Black/AA',3:'Am Indian',5:'Asian',50:'Other'}.items()], format_func=lambda x: x[0])
            p_marital = st.selectbox("Marital Status", [(v,k) for k,v in {1:'Married',2:'Widowed',3:'Divorced',4:'Separated',5:'Never married'}.items()], format_func=lambda x: x[0])
            p_hispanic = st.selectbox("Hispanic", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- VITALS ---
            st.markdown(f"#### 🩺 Vitals & Physical")
            p_bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.5)
            p_bpsys = st.slider("Systolic BP", 80, 220, 130)
            p_bpdias = st.slider("Diastolic BP", 40, 140, 80)
            p_hrate = st.slider("Heart Rate", 40, 140, 72)

            # --- GENETICS ---
            st.markdown(f"#### 🧬 Genetics")
            p_apoe = st.selectbox("APOE e4 Alleles", [('0 copies',0),('1 copy',1),('2 copies',2)], format_func=lambda x: x[0])

            # --- CARDIOVASCULAR ---
            st.markdown(f"#### ❤️ Cardiovascular")
            p_htn = st.selectbox("Hypertension", [('No',0),('Yes',1)], format_func=lambda x: x[0], key='p_htn')
            p_cvhatt = st.selectbox("Heart Attack History", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_stroke = st.selectbox("Stroke History", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_tia = st.selectbox("TIA History", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_afib = st.selectbox("Atrial Fibrillation", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_chf = st.selectbox("Congestive Heart Failure", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_bypass = st.selectbox("Bypass Surgery", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_angina = st.selectbox("Angina", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- RENAL / GENITOURINARY ---
            st.markdown(f"#### 🫘 Renal-Genitourinary")
            p_incontu = st.selectbox("Urinary Incontinence", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_incontf = st.selectbox("Fecal Incontinence", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- BLOOD / CELL PRODUCTION ---
            st.markdown(f"#### 🩸 Blood & Cell Production")
            p_b12 = st.selectbox("B12 Deficiency", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_thyroid = st.selectbox("Thyroid Disease", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- SURGICAL ---
            st.markdown(f"#### 🏥 Surgical History")
            p_angiop = st.selectbox("Angioplasty/Stent", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_pacemaker = st.selectbox("Pacemaker", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_hvalve = st.selectbox("Heart Valve Replacement", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- OTHER MEDICAL ---
            st.markdown(f"#### 💊 Other Medical Conditions")
            p_diabetes = st.selectbox("Diabetes", [('No',0),('Yes',1)], format_func=lambda x: x[0], key='p_diab')
            p_tbi = st.selectbox("Head Injury (TBI)", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_dep = st.selectbox("Depression (past 2yr)", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_cancer = st.selectbox("Cancer History", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_seizures = st.selectbox("Seizures", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_pd = st.selectbox("Parkinson Disease", [('No',0),('Yes',1)], format_func=lambda x: x[0])

            # --- LIFESTYLE ---
            st.markdown(f"#### 🚬 Lifestyle")
            p_tobac = st.selectbox("Tobacco Use (100+ cigs)", [('No',0),('Yes',1)], format_func=lambda x: x[0])
            p_alcohol = st.selectbox("Alcohol Abuse", [('No',0),('Yes',1)], format_func=lambda x: x[0], key='p_alc')

            # --- COGNITIVE (kept per professor) ---
            st.markdown(f"#### 🧠 Cognitive Measures")
            p_cdrsum = st.slider("CDR Sum of Boxes", 0.0, 18.0, 0.5, 0.5)
            p_cdrglob = st.selectbox("CDR Global", [0, 0.5, 1, 2, 3])
            p_memory = st.selectbox("CDR Memory", [0, 0.5, 1, 2, 3])
            p_orient = st.selectbox("CDR Orientation", [0, 0.5, 1, 2, 3])
            p_judgment = st.selectbox("CDR Judgment", [0, 0.5, 1, 2, 3])

            predict_btn = st.button("🔮 Predict Cognitive Status", type="primary", use_container_width=True)

        # --- RESULTS ---
        with col_right:
            if predict_btn:
                # Build feature vector
                sentinel = [888, 888.8, 999, 995, -4, -4.4, 88]
                df_all = df[df['NORMCOG'].notna()].copy()
                fv = {}
                for feat in lasso_features:
                    if feat in df_all.columns:
                        med = df_all[feat].replace(sentinel, np.nan).median()
                        fv[feat] = med if not pd.isna(med) else 0
                    else:
                        fv[feat] = 0

                # Override with user inputs
                input_map = {
                    'NACCAGE': p_age, 'SEX': p_sex[1], 'EDUC': p_educ, 'RACE': p_race[1],
                    'MARISTAT': p_marital[1], 'HISPANIC': p_hispanic[1],
                    'NACCBMI': p_bmi, 'BPSYS': p_bpsys, 'BPDIAS': p_bpdias, 'HRATE': p_hrate,
                    'NACCNE4S': p_apoe[1],
                    'HYPERTEN': p_htn[1], 'CVHATT': p_cvhatt[1], 'CBSTROKE': p_stroke[1],
                    'CBTIA': p_tia[1], 'CVAFIB': p_afib[1], 'CVCHF': p_chf[1],
                    'CVBYPASS': p_bypass[1], 'ANGINA': p_angina[1],
                    'INCONTU': p_incontu[1], 'INCONTF': p_incontf[1],
                    'B12DEF': p_b12[1], 'THYROID': p_thyroid[1],
                    'ANGIOPCI': p_angiop[1], 'PACEMAKE': p_pacemaker[1], 'HVALVE': p_hvalve[1],
                    'DIABETES': p_diabetes[1], 'NACCTBI': p_tbi[1], 'DEP2YRS': p_dep[1],
                    'CANCER': p_cancer[1], 'SEIZURES': p_seizures[1], 'PD': p_pd[1],
                    'TOBAC100': p_tobac[1], 'ALCOHOL': p_alcohol[1],
                    'CDRSUM': p_cdrsum, 'CDRGLOB': p_cdrglob,
                    'MEMORY': p_memory, 'ORIENT': p_orient, 'JUDGMENT': p_judgment,
                }
                for feat, val in input_map.items():
                    if feat in fv:
                        fv[feat] = val

                X_pred = pd.DataFrame([fv])[lasso_features]
                prob_normal = xgb_model.predict_proba(X_pred)[0, 1]
                prob_impaired = 1 - prob_normal

                if prob_normal >= 0.7:
                    tier, tc = 'LIKELY NORMAL', UTRGV_GREEN
                elif prob_normal >= 0.4:
                    tier, tc = 'UNCERTAIN', UTRGV_ORANGE
                else:
                    tier, tc = 'LIKELY IMPAIRED', '#D32F2F'

                pop_normal = df[df['NORMCOG'].notna()]['NORMCOG'].mean()

                # Display results
                st.markdown(f"""
                <div style="background:white;border:2px solid {tc};border-radius:12px;padding:30px;text-align:center;margin-bottom:20px;">
                    <p style="color:#6B6B6B;font-size:13px;text-transform:uppercase;font-weight:600;margin:0;">Probability of Normal Cognition</p>
                    <p style="font-size:56px;font-weight:800;color:{tc};margin:5px 0;line-height:1;">{prob_normal:.1%}</p>
                    <p style="font-size:18px;font-weight:700;color:{tc};margin:5px 0;">{tier}</p>
                    <p style="color:#6B6B6B;font-size:13px;margin:5px 0;">
                        Prediction: <strong style="color:{UTRGV_GREEN if prob_normal>=0.5 else '#D32F2F'}">
                        {'Normal Cognition' if prob_normal>=0.5 else 'Cognitive Impairment'}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Population comparison
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.metric("This Patient", f"{prob_normal:.1%}")
                with comp_col2:
                    st.metric("Population Average", f"{pop_normal:.1%}")
                ratio = prob_normal / pop_normal if pop_normal > 0 else 0
                direction = "higher" if prob_normal > pop_normal else "lower"
                st.markdown(f"**{ratio:.1f}x {direction}** than population average")

                # SHAP
                try:
                    import shap
                    exp = shap.TreeExplainer(xgb_model)
                    sv = exp.shap_values(X_pred)
                    sdf = pd.DataFrame({'Feature':lasso_features,'SHAP':sv[0],'Abs':np.abs(sv[0]),'Value':X_pred.iloc[0].values})
                    sdf = sdf.sort_values('Abs',ascending=False).head(15)
                    sdf['Label'] = sdf['Feature'].apply(get_label)
                    sdf['ValLabel'] = sdf.apply(lambda r: get_val_label(r['Feature'],r['Value']),axis=1)
                    sdf['Display'] = sdf['Label'] + ' = ' + sdf['ValLabel']
                    sdf = sdf.sort_values('SHAP',ascending=True)

                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(x=sdf['SHAP'],y=sdf['Display'],orientation='h',
                        marker_color=[UTRGV_GREEN if v>0 else UTRGV_ORANGE for v in sdf['SHAP']]))
                    fig_shap.update_layout(title="What's Driving This Prediction?",
                        xaxis_title='SHAP Value (green = supports normal, orange = supports impaired)',
                        plot_bgcolor='white',paper_bgcolor='white',height=480,
                        margin=dict(l=220,r=20,t=40,b=40))
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP analysis unavailable: {e}")

                # What-If Scenarios
                st.markdown("#### 🔄 What-If Scenarios")
                st.caption("Toggle each risk factor to see how the prediction changes")
                whatif_feats = {
                    'DIABETES':('Diabetes',p_diabetes[1]),'HYPERTEN':('Hypertension',p_htn[1]),
                    'DEP2YRS':('Depression',p_dep[1]),'NACCTBI':('Head Injury',p_tbi[1]),
                    'CANCER':('Cancer',p_cancer[1]),'CVHATT':('Heart Attack',p_cvhatt[1]),
                    'CBSTROKE':('Stroke',p_stroke[1]),'TOBAC100':('Tobacco',p_tobac[1]),
                }
                for feat, (label, curr) in whatif_feats.items():
                    if feat in fv:
                        tv = fv.copy()
                        tv[feat] = 1 - curr
                        tp = xgb_model.predict_proba(pd.DataFrame([tv])[lasso_features])[0, 1]
                        diff = tp - prob_normal
                        if abs(diff) > 0.001:
                            curr_l = 'Yes' if curr==1 else 'No'
                            tog_l = 'Yes' if (1-curr)==1 else 'No'
                            arrow = '↑' if diff > 0 else '↓'
                            color = UTRGV_GREEN if diff > 0 else '#D32F2F'
                            st.markdown(f"**{label}:** {curr_l} → {tog_l} &nbsp; "
                                       f"<span style='color:{color};font-weight:700;'>{arrow} {abs(diff):.1%}</span> "
                                       f"<span style='color:#999;font-size:12px;'>(normal prob: {tp:.1%})</span>",
                                       unsafe_allow_html=True)

                # Patient Summary
                st.markdown("#### 📋 Patient Input Summary")
                summary_data = {
                    'Age':p_age, 'Sex':p_sex[0], 'Education':f'{p_educ} yrs', 'Race':p_race[0],
                    'Marital':p_marital[0], 'BMI':p_bmi, 'Systolic BP':p_bpsys,
                    'APOE e4':p_apoe[0], 'CDR Sum':p_cdrsum, 'CDR Global':p_cdrglob,
                    'Diabetes':'Yes' if p_diabetes[1]==1 else 'No',
                    'Hypertension':'Yes' if p_htn[1]==1 else 'No',
                    'Depression':'Yes' if p_dep[1]==1 else 'No',
                    'Cancer':'Yes' if p_cancer[1]==1 else 'No',
                    'Stroke':'Yes' if p_stroke[1]==1 else 'No',
                }
                st.write(" | ".join([f"**{k}:** {v}" for k,v in summary_data.items()]))

            else:
                st.markdown("""
                <div style="background:white;border:1px solid #E0E0E0;border-radius:12px;padding:60px;text-align:center;">
                    <h3 style="color:#2D2D2D;">Cognitive Status Prediction</h3>
                    <p style="color:#6B6B6B;">Enter patient parameters on the left and click <strong>Predict</strong></p>
                    <p style="color:#999;font-size:12px;">Groups: Demographics • Vitals • Genetics • Cardiovascular • Renal-GU • Blood • Surgical • Medical • Lifestyle • Cognitive</p>
                </div>
                """, unsafe_allow_html=True)
