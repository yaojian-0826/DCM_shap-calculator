import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
import xgboost as xgb
import io
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             classification_report, accuracy_score)

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# ══════════════════════════════════════════
#  Fix Chinese font display in matplotlib
# ══════════════════════════════════════════
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════
#  Language Dictionary (i18n)
# ══════════════════════════════════════════
LANG = {
    "en": {
        # Navigation
        "nav_predict": "Predict & Explain",
        "nav_eval": "Model Evaluation", 
        "nav_global": "Global SHAP",
        "nav_samples": "Validation Samples",
        "nav_info": "Model Info",
        "nav_samples_train": "Training samples",
        "nav_samples_val": "Validation samples",
        "nav_features": "Features",
        "nav_algorithm": "Algorithm",
        "nav_base_value": "Base value",
        
        # Header
        "header_title": "XGBoost + SHAP Clinical Calculator",
        "header_desc": "Malignant Arrhythmia Risk Prediction for DCM Patients",
        
        # Page 1: Predict
        "p1_title": "Enter Patient Data",
        "p1_desc": "Input clinical parameters for a new patient. Values will be standardized automatically.",
        "p1_continuous": "Continuous Variables",
        "p1_binary": "Binary Variables",
        "p1_binary_hint": "0=No, 1=Yes",
        "p1_categorical": "Categorical Variables",
        "p1_btn_run": "Run Prediction & SHAP",
        "p1_result_high": "High Risk",
        "p1_result_low": "Low Risk",
        "p1_result_high_desc": "The model predicts high risk of malignant arrhythmia.",
        "p1_result_low_desc": "The model predicts low risk of malignant arrhythmia.",
        "p1_prob_case": "P(Case)",
        "p1_prob_control": "P(Control)",
        "p1_feat_values": "Feature Values (Standardized)",
        "p1_feat_raw": "Raw Value",
        "p1_feat_std": "Standardized",
        
        # SHAP
        "shap_title": "SHAP Explanation",
        "shap_waterfall": "Waterfall Plot",
        "shap_force": "Force Plot",
        "shap_force_desc": "Red = increases Case risk, Green = decreases",
        "shap_breakdown": "Top 10 SHAP Feature Contributions",
        "shap_value": "Value (std)",
        "shap_shap": "SHAP Value",
        "shap_effect_pos": "Increases risk",
        "shap_effect_neg": "Decreases risk",
        
        # Page 2: Evaluation
        "p2_title": "Model Performance on Validation Set",
        "p2_auc": "AUC-ROC",
        "p2_acc": "Accuracy",
        "p2_precision": "Precision (Case)",
        "p2_recall": "Recall (Case)",
        "p2_f1": "F1 (Case)",
        "p2_cm": "Confusion Matrix",
        "p2_roc": "ROC Curve",
        "p2_report": "Full Classification Report",
        "p2_pred": "Predicted",
        "p2_true": "True",
        
        # Page 3: Global SHAP
        "p3_title": "Global SHAP Analysis",
        "p3_desc": "SHAP values computed on the validation set",
        "p3_importance": "Feature Importance Ranking",
        "p3_bar": "SHAP Bar Plot",
        "p3_bee": "SHAP Beeswarm Plot",
        "p3_dep": "SHAP Dependence Plot",
        "p3_select_feat": "Select feature:",
        
        # Page 4: Samples
        "p4_title": "Explore Validation Samples",
        "p4_desc": "Browse any of the validation samples.",
        "p4_sample_idx": "Sample index",
        "p4_showing": "Showing sample",
        "p4_predicted": "Predicted",
        "p4_true_label": "True label",
        "p4_correct": "Correct",
        "p4_wrong": "Wrong",
        "p4_feat_vals": "Feature values (standardized)",
        
        # Common
        "case": "Case",
        "control": "Control",
        "built_with": "Built with Streamlit + XGBoost + SHAP",
        "language": "Language",
        
        # TNEA
        "tnea_label": "TNEA (Total ECG Abnormalities)",
        "tnea_count": "abnormalities",
        
        # Infection
        "infection_no": "0 (No)",
        "infection_yes": "1 (Yes)",
    },
    "zh": {
        # Navigation
        "nav_predict": "预测与解释",
        "nav_eval": "模型评估", 
        "nav_global": "全局SHAP",
        "nav_samples": "验证样本",
        "nav_info": "模型信息",
        "nav_samples_train": "训练样本",
        "nav_samples_val": "验证样本",
        "nav_features": "特征数",
        "nav_algorithm": "算法",
        "nav_base_value": "基准值",
        
        # Header
        "header_title": "XGBoost + SHAP 临床计算器",
        "header_desc": "扩张型心肌病患者恶性心律失常风险预测",
        
        # Page 1: Predict
        "p1_title": "输入患者数据",
        "p1_desc": "输入新患者的临床参数，数值将自动标准化",
        "p1_continuous": "连续变量",
        "p1_binary": "二分类变量",
        "p1_binary_hint": "0=否, 1=是",
        "p1_categorical": "多分类变量",
        "p1_btn_run": "运行预测与SHAP解释",
        "p1_result_high": "高风险",
        "p1_result_low": "低风险",
        "p1_result_high_desc": "模型预测该患者恶性心律失常风险较高",
        "p1_result_low_desc": "模型预测该患者恶性心律失常风险较低",
        "p1_prob_case": "病例组概率",
        "p1_prob_control": "对照组概率",
        "p1_feat_values": "特征值（标准化后）",
        "p1_feat_raw": "原始值",
        "p1_feat_std": "标准化值",
        
        # SHAP
        "shap_title": "SHAP解释",
        "shap_waterfall": "瀑布图",
        "shap_force": "力图",
        "shap_force_desc": "红色=增加病例风险，绿色=降低病例风险",
        "shap_breakdown": "前10个SHAP特征贡献",
        "shap_value": "值(标准化)",
        "shap_shap": "SHAP值",
        "shap_effect_pos": "增加风险",
        "shap_effect_neg": "降低风险",
        
        # Page 2: Evaluation
        "p2_title": "验证集模型性能",
        "p2_auc": "AUC-ROC",
        "p2_acc": "准确率",
        "p2_precision": "精确率(病例)",
        "p2_recall": "召回率(病例)",
        "p2_f1": "F1分数(病例)",
        "p2_cm": "混淆矩阵",
        "p2_roc": "ROC曲线",
        "p2_report": "完整分类报告",
        "p2_pred": "预测值",
        "p2_true": "真实值",
        
        # Page 3: Global SHAP
        "p3_title": "全局SHAP分析",
        "p3_desc": "基于验证集计算的SHAP值",
        "p3_importance": "特征重要性排名",
        "p3_bar": "SHAP条形图",
        "p3_bee": "SHAP蜂群图",
        "p3_dep": "SHAP依赖图",
        "p3_select_feat": "选择特征:",
        
        # Page 4: Samples
        "p4_title": "探索验证样本",
        "p4_desc": "浏览验证集中的任意样本",
        "p4_sample_idx": "样本索引",
        "p4_showing": "当前显示样本",
        "p4_predicted": "预测结果",
        "p4_true_label": "真实标签",
        "p4_correct": "正确",
        "p4_wrong": "错误",
        "p4_feat_vals": "特征值（标准化）",
        
        # Common
        "case": "病例组",
        "control": "对照组",
        "built_with": "基于 Streamlit + XGBoost + SHAP 构建",
        "language": "语言",
        
        # TNEA
        "tnea_label": "TNEA (心电图异常总数)",
        "tnea_count": "项异常",
        
        # Infection
        "infection_no": "0 (无)",
        "infection_yes": "1 (有)",
    }
}

# Feature labels in both languages
FEATURE_LABELS = {
    "en": {
        "TNEA": "TNEA (Total ECG Abnormalities)",
        "Aortic Annulus": "Aortic Annulus (mm)",
        "LDH": "LDH (U/L)",
        "Normalized Myoglobin": "Normalized Myoglobin",
        "Serum Potassium": "Serum Potassium (mmol/L)",
        "CK": "CK (U/L)",
        "MVPG": "MVPG (mmHg)",
        "Infection": "Infection",
        "Ⅲ-AAD": "III-AAD",
        "Dopamine": "Dopamine Use",
        "APB": "APB (Atrial Premature Beats)",
        "VPB": "VPB (Ventricular Premature Beats)",
        "VER": "VER (Ventricular Escape Rhythm)",
        "Sinus Bradycardia": "Sinus Bradycardia",
        "IVAC": "IVAC (IV Antiarrhythmic)",
        "Syncope": "Syncope",
        "NYHA Ⅳ": "NYHA Class IV"
    },
    "zh": {
        "TNEA": "TNEA (心电图异常总数)",
        "Aortic Annulus": "主动脉瓣环直径 (mm)",
        "LDH": "乳酸脱氢酶 (U/L)",
        "Normalized Myoglobin": "标准化肌红蛋白",
        "Serum Potassium": "血清钾 (mmol/L)",
        "CK": "肌酸激酶 (U/L)",
        "MVPG": "平均跨瓣压差 (mmHg)",
        "Infection": "感染",
        "Ⅲ-AAD": "三度房室传导阻滞",
        "Dopamine": "多巴胺使用",
        "APB": "房性早搏",
        "VPB": "室性早搏",
        "VER": "室性逸搏心律",
        "Sinus Bradycardia": "窦性心动过缓",
        "IVAC": "静脉抗心律失常药",
        "Syncope": "晕厥",
        "NYHA Ⅳ": "NYHA心功能IV级"
    }
}

# ══════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════
st.set_page_config(
    page_title="XGBoost + SHAP Calculator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 60%, #818cf8 100%);
    color: white;
    padding: 2rem 2.5rem 1.6rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(79,70,229,0.22);
}
.hero h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 0.3rem; letter-spacing: -0.5px; }
.hero p  { font-size: 0.95rem; margin: 0; opacity: 0.85; }

.pred-case    { background:#fef2f2; border:2px solid #fca5a5; border-radius:12px; padding:1.2rem 1.5rem; }
.pred-control { background:#f0fdf4; border:2px solid #86efac; border-radius:12px; padding:1.2rem 1.5rem; }
.pred-label-case    { font-size:1.6rem; font-weight:800; color:#dc2626; }
.pred-label-control { font-size:1.6rem; font-weight:800; color:#059669; }

.metric-row { display:flex; gap:12px; flex-wrap:wrap; margin:0.5rem 0 1rem; }
.mcard {
    flex:1; min-width:100px; background:white;
    border:1.5px solid #e2e8f0; border-radius:10px;
    padding:0.9rem 1rem; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
}
.mval  { font-size:1.45rem; font-weight:800; color:#4f46e5; }
.mlbl  { font-size:0.74rem; color:#64748b; margin-top:2px; }

.shap-pos { color:#dc2626; font-weight:700; }
.shap-neg { color:#059669; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  Data & Model Configuration
# ══════════════════════════════════════════
TRAIN_FILE = "shap解释训练集.xlsx"
VAL_FILE   = "shap解释测试集.xlsx"
TARGET_COL = "组别"
TARGET_MAP = {"case": 1, "control": 0}

# ══════════════════════════════════════════
#  Model Training (Cached)
# ══════════════════════════════════════════
@st.cache_resource(show_spinner="Training XGBoost model...")
def load_model():
    train_data = pd.read_excel(TRAIN_FILE, index_col=0)
    val_data = pd.read_excel(VAL_FILE, index_col=0)
    
    X_train = train_data.drop(columns=[TARGET_COL])
    y_train = train_data[TARGET_COL].map(TARGET_MAP)
    X_val = val_data.drop(columns=[TARGET_COL])
    y_val = val_data[TARGET_COL].map(TARGET_MAP)
    
    feature_names = X_train.columns.tolist()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=355,
        learning_rate=0.136005723363487,
        max_depth=1,
        min_child_weight=3,
        gamma=4.44151101866737,
        colsample_bytree=0.498135434091091,
        subsample=0.627589079085737,
        random_state=123,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_df, y_train)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_val_df)
    base_value = float(explainer.expected_value)
    
    return (xgb_model, explainer, scaler, feature_names, 
            X_train_df, y_train, X_val_df, y_val, shap_values, base_value)

def get_shap_single(explainer, X_row):
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        return sv[1][0] if len(sv) == 2 else sv[0]
    return sv[0] if sv.ndim > 1 else sv

# ══════════════════════════════════════════
#  Custom Force Plot (with language support)
# ══════════════════════════════════════════
def plot_force_plot_matplotlib(base_value, shap_values, features, feature_names, lang="en"):
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    n_display = min(10, len(feature_names))
    top_idx = sorted_idx[:n_display]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    y_pos = np.arange(n_display)
    colors = ['#ef4444' if shap_values[i] > 0 else '#10b981' for i in top_idx]
    
    bars = ax.barh(y_pos, [shap_values[i] for i in top_idx], 
                   color=colors, height=0.6, alpha=0.85)
    
    FL = FEATURE_LABELS[lang]
    display_names = []
    for i in top_idx:
        fname = feature_names[i]
        label = FL.get(fname, fname)
        if len(label) > 35:
            label = label[:32] + "..."
        display_names.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    
    xlabel = 'SHAP Value (contribution to prediction)' if lang == "en" else 'SHAP值（对预测的贡献）'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.axvline(0, color='black', linewidth=1)
    
    for bar, i in zip(bars, top_idx):
        width = bar.get_width()
        label_x = width + 0.02 if width >= 0 else width - 0.02
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{shap_values[i]:+.4f}\n(val={features[i]:.2f})', 
                va='center', ha=ha, fontsize=8)
    
    prediction = base_value + np.sum(shap_values)
    title = 'SHAP Force Plot' if lang == "en" else 'SHAP力图'
    ax.set_title(f'{title}\nBase: {base_value:.4f} -> Prediction: {prediction:.4f}',
                 fontsize=12, fontweight='bold', pad=15)
    
    from matplotlib.patches import Patch
    if lang == "en":
        legend_elements = [Patch(facecolor='#ef4444', label='Increases Case risk'),
                           Patch(facecolor='#10b981', label='Decreases Case risk')]
    else:
        legend_elements = [Patch(facecolor='#ef4444', label='增加病例风险'),
                           Patch(facecolor='#10b981', label='降低病例风险')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════
(xgb_model, explainer, scaler, feature_names, 
 X_train, y_train, X_val, y_val, shap_values_all, base_val) = load_model()

# ══════════════════════════════════════════
#  Language selector
# ══════════════════════════════════════════
lang_col1, lang_col2 = st.columns([6, 1])
with lang_col2:
    lang = st.selectbox(
        "Language",
        ["en", "zh"],
        format_func=lambda x: "English" if x == "en" else "中文",
        label_visibility="collapsed"
    )

T = LANG[lang]
FL = FEATURE_LABELS[lang]

# ══════════════════════════════════════════
#  Header
# ══════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>🎯 {T['header_title']}</h1>
  <p>{T['header_desc']}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## 🧭 {T['nav_predict'].split()[0]}")
    page = st.radio(
        "", 
        [f"🔮 {T['nav_predict']}", f"📊 {T['nav_eval']}",
         f"🌐 {T['nav_global']}", f"📋 {T['nav_samples']}"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**{T['nav_info']}**")
    st.caption(f"{T['nav_samples_train']}: **{len(X_train)}**")
    st.caption(f"{T['nav_samples_val']}: **{len(X_val)}**")
    st.caption(f"{T['nav_features']}: **{len(feature_names)}**")
    st.caption(f"{T['nav_algorithm']}: **XGBoost**")
    st.caption(f"{T['nav_base_value']}: **{base_val:.4f}**")
    st.markdown("---")
    st.caption(T['built_with'])

# ══════════════════════════════════════════
#  PAGE 1 — Predict & Explain
# ══════════════════════════════════════════
if page.startswith("🔮"):
    st.subheader(f"🧪 {T['p1_title']}")
    st.caption(T['p1_desc'])
    
    col_left, col_mid, col_right = st.columns(3, gap="large")
    
    with col_left:
        st.markdown(f"##### {T['p1_continuous']}")
        aortic = st.number_input(FL["Aortic Annulus"], value=23.0, step=0.1)
        ldh = st.number_input(FL["LDH"], value=250.0, step=1.0)
        myoglobin = st.number_input(FL["Normalized Myoglobin"], value=0.0, step=0.01, format="%.4f")
        potassium = st.number_input(FL["Serum Potassium"], value=4.0, step=0.1)
        ck = st.number_input(FL["CK"], value=100.0, step=1.0)
        mvpg = st.number_input(FL["MVPG"], value=10.0, step=0.1)
    
    with col_mid:
        st.markdown(f"##### {T['p1_categorical']}")
        # TNEA - Categorical (0-14, extendable)
        tnea = st.number_input(
            FL["TNEA"],
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help=f"Enter the total number of ECG abnormalities (0-14+)" if lang == "en" else "输入心电图异常总数（0-14项及以上）"
        )
        
        st.markdown(f"##### {T['p1_binary']}")
        # Infection - Binary (0 or 1)
        infection = st.selectbox(
            FL["Infection"],
            [0, 1],
            format_func=lambda x: T['infection_no'] if x == 0 else T['infection_yes']
        )
    
    with col_right:
        st.markdown(f"##### {T['p1_binary']} ({T['p1_binary_hint']})")
        iii_aad = st.selectbox(FL["Ⅲ-AAD"], [0, 1])
        dopamine = st.selectbox(FL["Dopamine"], [0, 1])
        apb = st.selectbox(FL["APB"], [0, 1])
        vpb = st.selectbox(FL["VPB"], [0, 1])
        ver = st.selectbox(FL["VER"], [0, 1])
        sinus_brady = st.selectbox(FL["Sinus Bradycardia"], [0, 1])
        ivac = st.selectbox(FL["IVAC"], [0, 1])
        syncope = st.selectbox(FL["Syncope"], [0, 1])
        nyha4 = st.selectbox(FL["NYHA Ⅳ"], [0, 1])
    
    submitted = st.button(f"🚀 {T['p1_btn_run']}", type="primary", use_container_width=True)
    
    if submitted:
        raw_input = np.array([[
            tnea, aortic, ldh, myoglobin, potassium, ck, mvpg, infection,
            iii_aad, dopamine, apb, vpb, ver, sinus_brady, ivac, syncope, nyha4
        ]])
        
        scaled_input = scaler.transform(raw_input)
        sample_df = pd.DataFrame(scaled_input, columns=feature_names)
        
        pred_class = int(xgb_model.predict(sample_df)[0])
        pred_prob = xgb_model.predict_proba(sample_df)[0]
        
        col_res1, col_res2 = st.columns([1, 1.2], gap="large")
        
        with col_res1:
            if pred_class == 1:
                st.markdown(f"""
                <div class="pred-case">
                  <div class="pred-label-case">⚠️ {T['p1_result_high']}</div>
                  <div style="color:#7f1d1d;margin-top:4px;font-size:.9rem">
                    {T['p1_result_high_desc']}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-control">
                  <div class="pred-label-control">✅ {T['p1_result_low']}</div>
                  <div style="color:#14532d;margin-top:4px;font-size:.9rem">
                    {T['p1_result_low_desc']}
                  </div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric(T['p1_prob_case'], f"{pred_prob[1]*100:.1f}%")
            c2.metric(T['p1_prob_control'], f"{pred_prob[0]*100:.1f}%")
            
            fig_pb, ax = plt.subplots(figsize=(5, 1.6))
            labels = [T['control'], T['case']]
            bars = ax.barh(labels, [pred_prob[0], pred_prob[1]],
                          color=["#10b981","#ef4444"], height=0.5)
            ax.set_xlim(0, 1)
            prob_label = "Probability" if lang == "en" else "概率"
            ax.set_xlabel(prob_label)
            ax.axvline(0.5, color="gray", lw=1, ls="--")
            for b, v in zip(bars, [pred_prob[0], pred_prob[1]]):
                ax.text(v + 0.01, b.get_y() + b.get_height()/2,
                       f"{v*100:.1f}%", va="center", fontsize=10, fontweight="bold")
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_pb, use_container_width=True)
            plt.close(fig_pb)
        
        with col_res2:
            st.markdown(f"**📊 {T['p1_feat_values']}**")
            feat_display = pd.DataFrame({
                "Feature": [FL.get(f, f) for f in feature_names],
                T['p1_feat_raw']: raw_input[0],
                T['p1_feat_std']: scaled_input[0]
            })
            st.dataframe(feat_display.round(3), use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader(f"🔍 {T['shap_title']}")
        
        sv_s = get_shap_single(explainer, sample_df)
        dv_r = scaled_input[0]
        bv_r = round(base_val, 4)
        
        exp_obj = shap.Explanation(
            values=np.round(sv_s, 4), 
            base_values=bv_r,
            data=np.round(dv_r, 4), 
            feature_names=feature_names
        )
        
        col_wf, col_fp = st.columns(2, gap="large")
        
        with col_wf:
            st.markdown(f"**💧 {T['shap_waterfall']}**")
            fig_wf, _ = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(exp_obj, show=False, max_display=10)
            waterfall_title = "SHAP Waterfall (Your Patient)" if lang == "en" else "SHAP瀑布图（当前患者）"
            plt.title(waterfall_title, fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)
        
        with col_fp:
            st.markdown(f"**⚡ {T['shap_force']}**")
            st.caption(T['shap_force_desc'])
            fig_fp = plot_force_plot_matplotlib(bv_r, sv_s, dv_r, feature_names, lang)
            st.pyplot(fig_fp, use_container_width=True)
            plt.close(fig_fp)
        
        st.markdown(f"**📊 {T['shap_breakdown']}**")
        shap_breakdown = []
        for f, v, s in zip(feature_names, dv_r, sv_s):
            shap_breakdown.append({
                "Feature": FL.get(f, f),
                T['shap_value']: round(float(v), 4),
                T['shap_shap']: round(float(s), 4),
                "Effect": f"↑ {T['shap_effect_pos']}" if s >= 0 else f"↓ {T['shap_effect_neg']}"
            })
        shap_breakdown.sort(key=lambda x: abs(x[T['shap_shap']]), reverse=True)
        
        df_bd = pd.DataFrame(shap_breakdown[:10])
        def color_shap(val):
            if isinstance(val, float):
                return f"color: {'#dc2626' if val > 0 else '#059669'}; font-weight: bold"
            return ""
        
        st.dataframe(df_bd.style.applymap(color_shap, subset=[T['shap_shap']]),
                    use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
#  PAGE 2 — Model Evaluation
# ══════════════════════════════════════════
elif page.startswith("📊"):
    st.subheader(f"📈 {T['p2_title']}")
    
    y_pred = xgb_model.predict(X_val)
    y_prob = xgb_model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred) * 100
    rep = classification_report(y_val, y_pred, output_dict=True, 
                                target_names=[T['control'], T['case']])
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(T['p2_auc'], f"{auc:.3f}")
    m2.metric(T['p2_acc'], f"{acc:.1f}%")
    m3.metric(T['p2_precision'], f"{rep[T['case']]['precision']*100:.1f}%")
    m4.metric(T['p2_recall'], f"{rep[T['case']]['recall']*100:.1f}%")
    m5.metric(T['p2_f1'], f"{rep[T['case']]['f1-score']*100:.1f}%")
    
    st.divider()
    col_cm, col_roc = st.columns(2, gap="large")
    
    with col_cm:
        st.markdown(f"**{T['p2_cm']}**")
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=[T['control'], T['case']],
                   yticklabels=[T['control'], T['case']], ax=ax, annot_kws={"size":14})
        ax.set_title(T['p2_cm'], fontsize=13, fontweight="bold")
        ax.set_xlabel(T['p2_pred']); ax.set_ylabel(T['p2_true'])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col_roc:
        st.markdown(f"**{T['p2_roc']}**")
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {auc:.3f}")
        ax2.plot([0,1],[0,1], "navy", lw=1.5, ls="--", label="Random" if lang == "en" else "随机")
        ax2.fill_between(fpr, tpr, alpha=0.08, color="darkorange")
        fpr_label = "False Positive Rate" if lang == "en" else "假阳性率"
        tpr_label = "True Positive Rate" if lang == "en" else "真阳性率"
        ax2.set_xlabel(fpr_label); ax2.set_ylabel(tpr_label)
        ax2.set_title(T['p2_roc'], fontsize=13, fontweight="bold")
        ax2.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    
    st.divider()
    st.markdown(f"**{T['p2_report']}**")
    rep_df = pd.DataFrame(rep).transpose().round(3)
    st.dataframe(rep_df, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE 3 — Global SHAP
# ══════════════════════════════════════════
elif page.startswith("🌐"):
    st.subheader(f"🌐 {T['p3_title']}")
    st.caption(f"{T['p3_desc']} ({len(X_val)} samples)" if lang == "en" else f"{T['p3_desc']}（{len(X_val)}个样本）")
    
    mean_abs = np.abs(shap_values_all).mean(0)
    imp_df = pd.DataFrame({
        "Feature": [FL.get(f, f) for f in feature_names],
        "Mean |SHAP|": np.round(mean_abs, 4),
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    
    col_tbl, col_bar = st.columns([1, 1.4], gap="large")
    
    with col_tbl:
        st.markdown(f"**{T['p3_importance']}**")
        st.dataframe(imp_df, use_container_width=True)
    
    with col_bar:
        st.markdown(f"**{T['p3_bar']}**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_all, X_val, plot_type="bar", show=False, color="#5865F2")
        bar_title = "Feature Importance (SHAP)" if lang == "en" else "特征重要性（SHAP）"
        plt.title(bar_title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    st.divider()
    
    col_bee, col_dep = st.columns(2, gap="large")
    
    with col_bee:
        st.markdown(f"**{T['p3_bee']}**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_all, X_val, show=False)
        bee_title = "Feature Impact Distribution" if lang == "en" else "特征影响分布"
        plt.title(bee_title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col_dep:
        st.markdown(f"**{T['p3_dep']}**")
        top_idx = int(np.argsort(mean_abs)[-1])
        top_feat = feature_names[top_idx]
        sel_feat = st.selectbox(T['p3_select_feat'], feature_names, index=top_idx)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(sel_feat, shap_values_all, X_val,
                            interaction_index="auto", show=False, ax=ax)
        dep_title = f"Dependence: {FL.get(sel_feat, sel_feat)}" if lang == "en" else f"依赖图: {FL.get(sel_feat, sel_feat)}"
        ax.set_title(dep_title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ══════════════════════════════════════════
#  PAGE 4 — Validation Samples
# ══════════════════════════════════════════
elif page.startswith("📋"):
    st.subheader(f"📋 {T['p4_title']}")
    st.caption(T['p4_desc'])
    
    col_ctrl, _ = st.columns([1, 2])
    with col_ctrl:
        idx = st.slider(T['p4_sample_idx'], min_value=0, max_value=len(X_val)-1, value=0, step=1)
        st.caption(f"{T['p4_showing']} **#{idx}**")
    
    sv_s = shap_values_all[idx]
    dv_r = X_val.iloc[idx].values
    pred = int(xgb_model.predict(X_val.iloc[[idx]])[0])
    prob = xgb_model.predict_proba(X_val.iloc[[idx]])[0]
    true_l = int(y_val.iloc[idx])
    correct = pred == true_l
    
    col_info, col_feat = st.columns([1, 1.2], gap="large")
    
    with col_info:
        label_str = T['case'] if pred == 1 else T['control']
        true_str = T['case'] if true_l == 1 else T['control']
        correct_str = T['p4_correct'] if correct else T['p4_wrong']
        
        if pred == 1:
            st.markdown(f"""
            <div class="pred-case">
              <div class="pred-label-case">⚠️ {T['p4_predicted']}: {label_str}</div>
              <div style="color:#7f1d1d;margin-top:6px">
                {T['p4_true_label']}: <b>{true_str}</b> &nbsp; {"✅" if correct else "❌"} {correct_str}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-control">
              <div class="pred-label-control">✅ {T['p4_predicted']}: {label_str}</div>
              <div style="color:#14532d;margin-top:6px">
                {T['p4_true_label']}: <b>{true_str}</b> &nbsp; {"✅" if correct else "❌"} {correct_str}
              </div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric(T['p1_prob_case'], f"{prob[1]*100:.1f}%")
        c2.metric(T['p1_prob_control'], f"{prob[0]*100:.1f}%")
    
    with col_feat:
        st.markdown(f"**{T['p4_feat_vals']}**")
        feat_df = pd.DataFrame({
            "Feature": [FL.get(f, f) for f in feature_names],
            "Value": np.round(dv_r, 4),
            "SHAP": np.round(sv_s, 4),
        }).sort_values("SHAP", key=abs, ascending=False).reset_index(drop=True)
        feat_df.index += 1
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    exp_obj = shap.Explanation(
        values=np.round(sv_s, 4), 
        base_values=round(base_val, 4),
        data=np.round(dv_r, 4), 
        feature_names=feature_names
    )
    
    col_wf, col_fp = st.columns(2, gap="large")
    
    with col_wf:
        st.markdown(f"**💧 {T['shap_waterfall']} — Sample #{idx}**")
        fig, _ = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(exp_obj, show=False, max_display=10)
        plt.title(f"SHAP Waterfall (Sample #{idx})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with col_fp:
        st.markdown(f"**⚡ {T['shap_force']} — Sample #{idx}**")
        fig2 = plot_force_plot_matplotlib(base_val, sv_s, dv_r, feature_names, lang)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
