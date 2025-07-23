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

LIMIT_APPLICANTS = 5000  

def load_json_upload(uploaded_json, limit_applicants=False):
    data = json.load(uploaded_json)
    if limit_applicants:
        keys = list(data.keys())[:LIMIT_APPLICANTS] if len(data) > LIMIT_APPLICANTS else list(data.keys())
        data = {k: data[k] for k in keys}
    return data

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
    df_model = df.copy()
    # Remova TUDO o que é textual ou categórico largo (adaptar conforme necessidade)
    cols_to_drop = [
        'main_activities_job', 'technical_skills_job',
        'prospect_id', 'prospect_name', 'prospect_comment',
        'prospect_status', 'applicant_id', 'technical_knowledge',
        'client', 'job_id',    #não agregam e podem explodir NaNs
        'academic_level'
    ]
    # Só mantenha as colunas alvo e poucas cols categóricas de valor reduzido
    cols_to_keep = [c for c in df_model.columns if c not in cols_to_drop and c != 'is_hired']
    # Mantém apenas colunas com poucos valores únicos (<15)
    small_cat_cols = [
    c for c in cols_to_keep
    if (df_model[c].dtype == "object" and df_model[c].nunique() < 15) or
       (df_model[c].dtype != "object") or
       (c == 'area_of_expertise')  
    ]
    df_model = df_model[small_cat_cols + ['is_hired']]
    st.write("Colunas finais do model:", df_model.columns.tolist())
    st.write("Shape model final:", df_model.shape)
    st.write("Alvo is_hired distribuição:", df_model["is_hired"].value_counts())

    if df_model.empty or df_model['is_hired'].nunique() < 2:
        st.error("Não há dados suficientes ou as duas classes (contratado/não) na base filtrada. Impossível treinar/classificar.")
        return None, None, None, None

    X = df_model.drop(columns=['is_hired'])
    y = df_model['is_hired']

    # Defina variáveis categóricas e numéricas após o filtro
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
        ('classifier', RandomForestClassifier(n_estimators=7, class_weight='balanced', random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


    # Checagem mínima para evitar erro no confusion_matrix
    import numpy as np
    y_test_array = np.array(y_test).ravel()
    if len(np.unique(y_test_array)) < 2:
        st.error("Após split, só há uma classe na base de teste! Impossível gerar matriz de confusão.")
        return None, None, None, None

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred).ravel()
    cm = confusion_matrix(y_test_array, y_pred)
    report = classification_report(y_test_array, y_pred, output_dict=True)
    rf_model = clf.named_steps['classifier']

    # Names das features após preprocessamento
    numeric_features = num_cols.tolist()
    if len(cat_cols) > 0:
        cat_features_encoded = clf.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    else:
        cat_features_encoded = []
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
    fig, ax = plt.subplots(figsize=(7, top_n // 2 + 2))
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
        applicants_data = load_json_upload(applicants_json, limit_applicants=True)
        st.info(f"Utilizando {len(applicants_data)} registros de candidatos (limitado em {LIMIT_APPLICANTS} para performance).")

        df = combine_dataframes(vagas_data, prospects_data, applicants_data)
        st.success(f"Dados carregados com {df.shape[0]} candidatos e {df.shape[1]} atributos.")
        st.dataframe(df.head(30), use_container_width=True)
        # st.write("Tipo de colunas:", dict(zip(df.columns, df.dtypes)))

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
        applicants_data = load_json_upload(applicants_json, limit_applicants=True)
        st.info(f"Utilizando {len(applicants_data)} registros de candidatos (limitado em {LIMIT_APPLICANTS} para performance).")

        df = combine_dataframes(vagas_data, prospects_data, applicants_data)

        st.success("Arquivos carregados: Random Forest será treinada e avaliada nos próprios dados.")
        with st.spinner("Treinando modelo Random Forest..."):
            clf, cm, report, feature_importance_df = run_random_forest(df)
        
        if clf is None:
            st.stop()

        st.markdown("#### Desempenho do Modelo")
        if cm is not None:
            plot_confusion_matrix(cm)

        if report is not None:
            st.markdown("**Classification report:**")
            st.json({k: report[k] for k in ['0', '1', 'accuracy', 'macro avg', 'weighted avg']})

        st.markdown("#### Principais Fatores Decisivos (Top 15)")
        if feature_importance_df is not None:
            plot_feature_importances(feature_importance_df)
            st.info("A interpretação dos fatores principais pode indicar o 'perfil ideal' do candidato para as contratações históricas.")

        st.markdown("---")
        # Download do modelo treinado (opcional)
        model_filename = "random_forest_model.pkl"
        try:
            joblib.dump(clf, model_filename)
            with open(model_filename, "rb") as f:
                st.download_button("Baixar modelo Random Forest treinado (.pkl)", data=f, file_name=model_filename)
        except Exception as e:
            st.warning(f"Não foi possível salvar modelo: {e}")
    else:
        st.info("Faça upload dos 3 arquivos (.json) necessários para treinar e avaliar a Random Forest.")
