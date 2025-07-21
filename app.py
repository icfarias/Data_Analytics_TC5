import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(layout="wide", page_title="FIAP PÓS TECH – IA em Recrutamento", page_icon="🤖")

# -------- TÍTULO E LOGO --------
st.image("fiap_logo.png", width=240)
st.markdown("<h1 style='text-align: center; color: #0e4c92;'>FIAP PÓS TECH – IA em Recrutamento & Seleção</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #545454;'>TECH CHALLENGE 5</h2>", unsafe_allow_html=True)
st.markdown("---")

# -------- MENU LATERAL ---------
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Navegação",
    ["🏠 Introdução", "📊 Exploração e Personas", "🧩 Random Forest & Importâncias"],
    index=0
)

# -------- INTEGRANTES ---------
integrantes = [
    "Alexandre Barbosa",
    "Igor Calheiros de Farias",
    "João Paulo Machado",
    "Suellen dos Santos Rocha Godoi",
    "Thiago Moreira Dobbns"
]
integrantes = sorted(integrantes)
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Integrantes do Grupo")
integrantes_md = "".join([f"<div style='margin-bottom: -6px;'>👤 {nome}</div>" for nome in integrantes])
st.sidebar.markdown(integrantes_md, unsafe_allow_html=True)

# ---------- FUNÇÕES AUXILIARES ------------

def load_json_data(path):
    if not os.path.exists(path):
        st.warning(f"Arquivo '{path}' não foi encontrado.")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar {path}: {e}")
        return None

@st.cache_data
def combine_dataframes(jobs_data, prospects_data, applicants_data):
    if not all([jobs_data, prospects_data, applicants_data]):
        st.error("Erro: Um ou mais arquivos estão vazios ou ausentes.")
        return pd.DataFrame()
    all_records = []
    for job_id, job_info in jobs_data.items():
        informacoes_basicas = job_info.get('informacoes_basicas', {})
        perfil_vaga = job_info.get('perfil_vaga', {})
        job_record = {
            'job_id': job_id,
            'is_sap_job': informacoes_basicas.get('vaga_sap'),
            'client': informacoes_basicas.get('cliente'),
            'professional_level_required': perfil_vaga.get('nivel profissional'),
            'english_level_required': perfil_vaga.get('nivel_ingles'),
            'spanish_level_required': perfil_vaga.get('nivel_espanhol'),
            'main_activities_job': perfil_vaga.get('principais_atividades'),
            'technical_skills_job': perfil_vaga.get('competencia_tecnicas_e_comportamentais')
        }
        prospects = prospects_data.get(job_id, {}).get('prospects', [])
        for prospect in prospects:
            prospect_record = {
                'prospect_id': prospect.get('codigo'),
                'prospect_name': prospect.get('nome'),
                'prospect_comment': prospect.get('comentario'),
                'prospect_status': prospect.get('situacao_candidado'),
                'is_hired': 1 if 'contratado' in prospect.get('situacao_candidado', '').lower() else 0
            }
            applicant_id = prospect.get('codigo')
            applicant_info = applicants_data.get(str(applicant_id), {})
            formacao = applicant_info.get('formacao_e_idiomas', {})
            prof = applicant_info.get('informacoes_profissionais', {})
            applicant_record = {
                'applicant_id': applicant_id,
                'academic_level': formacao.get('nivel_academico'),
                'english_level_applicant': formacao.get('nivel_ingles'),
                'spanish_level_applicant': formacao.get('nivel_espanhol'),
                'technical_knowledge': prof.get('conhecimentos_tecnicos'),
                'area_of_expertise': prof.get('area_atuacao')
            }
            combined_record = {**job_record, **prospect_record, **applicant_record}
            all_records.append(combined_record)
    df = pd.DataFrame(all_records)
    return df

def run_random_forest(df):
    # Copia para não alterar original
    df_model = df.copy()
    cols_to_drop = [
        'main_activities_job', 'technical_skills_job',
        'prospect_id', 'prospect_name', 'prospect_comment',
        'prospect_status', 'applicant_id', 'technical_knowledge'
    ]
    df_model = df_model.drop(columns=[col for col in cols_to_drop if col in df_model.columns])
    X = df_model.drop(columns=['is_hired'])
    y = df_model['is_hired']
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(exclude='object').columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    # Importância das features
    rf_model = clf.named_steps['classifier']
    # Nomes das features após preprocessamento
    numeric_features = num_cols.tolist()
    cat_features_encoded = clf.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_feature_names = np.concatenate([numeric_features, cat_features_encoded])
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    return clf, cm, report, feature_importance_df

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (com balanceamento)")
    st.pyplot(fig)

def plot_feature_importances(feature_importance_df, top_n=15):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=feature_importance_df.head(top_n),
        x='importance',
        y='feature',
        palette='viridis',
        ax=ax
    )
    ax.set_title(f'Top {top_n} Features mais relevantes para contratação')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)

def personas_table(df):
    df_hired_3d = df[
        (df['is_hired'] == 1) &
        (df['professional_level_required'].notna()) &
        (df['english_level_applicant'].notna()) &
        (df['area_of_expertise'].notna()) &
        (df['english_level_applicant'].astype(str).str.strip() != '') &
        (df['area_of_expertise'].astype(str).str.strip() != '')
    ]
    personas_3d = (
        df_hired_3d
        .groupby(['professional_level_required', 'english_level_applicant', 'area_of_expertise'])
        .size()
        .reset_index(name='num_contratados')
        .sort_values(by='num_contratados', ascending=False)
        .head(5)
    )
    return personas_3d

# ------ PAGE 1: INTRODUÇÃO ------
if page == "🏠 Introdução":
    st.subheader("Introdução")
    st.markdown("""
Este MVP demonstra a aplicação de Inteligência Artificial no processo de recrutamento e seleção, integrando dados reais/simulados de candidatos, vagas e prospectos para treinamento e avaliação de algoritmos de classificação.  
O objetivo principal é mostrar, por meio da análise de dados e machine learning, como o perfil ideal do candidato e os fatores decisivos para contratação podem ser mapeados e visualizados no processo moderno de RH.
    """)
    st.markdown("---")
    st.info("Use o menu lateral para carregar e explorar os dados, visualizar personas e interpretar a IA!")

# ------ PAGE 2: EXPLORAÇÃO E PERSONAS ------
elif page == "📊 Exploração e Personas":
    st.title("📊 Exploração dos Dados de Candidatos")

    col_u, col_x, col_y = st.columns([2,1,2])
    with col_u:
        vaga_json = st.file_uploader("🗂️ Upload vagas.json", type="json", key="vagas")
        prospects_json = st.file_uploader("🗂️ Upload prospects.json", type="json", key="prospects")
        applicants_json = st.file_uploader("🗂️ Upload applicants.json", type="json", key="applicants")

    if vaga_json and prospects_json and applicants_json:
        vagas_data = json.load(vaga_json)
        prospects_data = json.load(prospects_json)
        applicants_data = json.load(applicants_json)
        df = combine_dataframes(vagas_data, prospects_data, applicants_data)

        st.success(f"Dados carregados com {df.shape[0]} candidatos e {df.shape[1]} atributos.")
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("#### Download do DataFrame compilado")
        st.download_button("Baixar CSV", data=df.to_csv(index=False), file_name="candidatos_combinados.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("### Top 5 Personas Contratados (3 fatores combinados)")
        df_personas = personas_table(df)
        st.table(df_personas)
        st.info("""
Cada linha representa um perfil de candidato mais frequentemente contratado — agrupando nível profissional requerido, nível de inglês informado e área de atuação.
        """)
    else:
        st.info("Faça upload dos 3 arquivos (.json) necessários para análise.")

# ------ PAGE 3: RANDOM FOREST, MATRIZ E FEAT IMPORTANCE ------
elif page == "🧩 Random Forest & Importâncias":
    st.title("🧩 Random Forest – Fatores Decisivos na Contratação")
    vaga_json = st.file_uploader("🗂️ Upload vagas.json", type="json", key="vagas_rf")
    prospects_json = st.file_uploader("🗂️ Upload prospects.json", type="json", key="prospects_rf")
    applicants_json = st.file_uploader("🗂️ Upload applicants.json", type="json", key="applicants_rf")

    if vaga_json and prospects_json and applicants_json:
        vagas_data = json.load(vaga_json)
        prospects_data = json.load(prospects_json)
        applicants_data = json.load(applicants_json)
        df = combine_dataframes(vagas_data, prospects_data, applicants_data)

        st.success("Arquivos carregados: Random Forest será treinada e avaliada nos próprios dados.")
        with st.spinner("Treinando modelo Random Forest..."):
            clf, cm, report, feature_importance_df = run_random_forest(df)
        st.markdown("#### Desempenho do Modelo")
        plot_confusion_matrix(cm)

        st.markdown("**Classification report:**")
        st.json({k: report[k] for k in ['0', '1', 'accuracy', 'macro avg', 'weighted avg']})

        st.markdown("#### Principais Fatores Decisivos (Top 15)")
        plot_feature_importances(feature_importance_df)
        st.info("A interpretação dos fatores principais pode indicar o 'perfil ideal' do candidato para as contratações históricas.")

        st.markdown("---")
        # Download do modelo treinado (opcional)
        model_filename = "random_forest_model.pkl"
        joblib.dump(clf, model_filename)
        with open(model_filename, "rb") as f:
            st.download_button("Baixar modelo Random Forest treinado (.pkl)", data=f, file_name=model_filename)
    else:
        st.info("Faça upload dos 3 arquivos (.json) necessários para treinar e avaliar a Random Forest.")
