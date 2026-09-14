import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Liver Patient Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

    /* Main title */
    .main-title {
        color: #2563EB;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Subtitle */
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    /* Section title */
    .section-title {
        color: #0F766E;
        font-size: 1.4rem;
        font-weight: 650;
    }

    /* Information card */
    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
        background-color: #F8FAFC;
        margin-bottom: 15px;
    }

    /* Small label */
    .small-label {
        color: #475569;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/liver_disease_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model tidak ditemukan. Pastikan file "
        "'models/liver_disease_model.joblib' tersedia."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🩺 Indian Liver Patient Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning application for liver patient classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.markdown("""
    **Machine Learning Model**

    Logistic Regression

    **Preprocessing**

    Min-Max Scaling

    **Dataset**

    Indian Liver Patient Dataset
    """)

    st.divider()

    st.markdown("""
    **Features**

    • Age  
    • Gender  
    • Total Bilirubin  
    • Direct Bilirubin  
    • Alkaline Phosphotase  
    • Alamine Aminotransferase  
    • Aspartate Aminotransferase  
    • Total Proteins  
    • Albumin  
    • Albumin and Globulin Ratio
    """)

    st.divider()

    st.caption("Machine Learning Project")


# ============================================================
# TABS
# ============================================================

tab_prediction, tab_dataset, tab_model = st.tabs([
    "🔍 Prediction",
    "📊 Dataset",
    "🤖 Model"
])


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab_prediction:

    st.markdown(
        '<div class="section-title">'
        'Patient Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Masukkan informasi pasien pada form berikut "
        "untuk melakukan klasifikasi."
    )

    st.info(
        "Silakan masukkan nilai pemeriksaan pasien sesuai "
        "data yang tersedia."
    )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Usia pasien dalam tahun."
        )

        gender = st.selectbox(
            "Gender",
            options=[1, 0],
            format_func=lambda x: (
                "Male" if x == 1 else "Female"
            ),
            help="1 = Male, 0 = Female."
        )

        total_bilirubin = st.number_input(
            "Total Bilirubin",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Kadar total bilirubin."
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            min_value=0.0,
            value=0.3,
            step=0.1,
            help="Kadar direct bilirubin."
        )

        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase",
            min_value=0.0,
            value=200.0,
            step=1.0,
            help="Nilai alkaline phosphotase."
        )

    with col2:

        alamine_aminotransferase = st.number_input(
            "Alamine Aminotransferase",
            min_value=0.0,
            value=30.0,
            step=1.0,
            help="Nilai alamine aminotransferase."
        )

        aspartate_aminotransferase = st.number_input(
            "Aspartate Aminotransferase",
            min_value=0.0,
            value=30.0,
            step=1.0,
            help="Nilai aspartate aminotransferase."
        )

        total_protiens = st.number_input(
            "Total Proteins",
            min_value=0.0,
            value=6.5,
            step=0.1,
            help="Total protein dalam darah."
        )

        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            value=3.5,
            step=0.1,
            help="Kadar albumin."
        )

        albumin_and_globulin_ratio = st.number_input(
            "Albumin and Globulin Ratio",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Rasio albumin terhadap globulin."
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Predict Patient Category",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        # Membuat DataFrame sesuai urutan feature saat training
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Total_Bilirubin": [total_bilirubin],
            "Direct_Bilirubin": [direct_bilirubin],
            "Alkaline_Phosphotase": [alkaline_phosphotase],
            "Alamine_Aminotransferase": [
                alamine_aminotransferase
            ],
            "Aspartate_Aminotransferase": [
                aspartate_aminotransferase
            ],
            "Total_Protiens": [total_protiens],
            "Albumin": [albumin],
            "Albumin_and_Globulin_Ratio": [
                albumin_and_globulin_ratio
            ]
        })

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        # Probability prediction
        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error(
                "⚠️ Hasil Prediksi: Kategori 1"
            )

            st.write(
                "Model mengklasifikasikan data pasien "
                "ke dalam **Kategori 1**."
            )

        elif prediction == 2:

            st.success(
                "✅ Hasil Prediksi: Kategori 2"
            )

            st.write(
                "Model mengklasifikasikan data pasien "
                "ke dalam **Kategori 2**."
            )

        else:

            st.warning(
                f"Hasil Prediksi: Kategori {prediction}"
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Prediction Probability'
            '</div>',
            unsafe_allow_html=True
        )

        probability_data = pd.DataFrame({
            "Kategori": [
                f"Kategori {category}"
                for category in classes
            ],
            "Probability": probabilities
        })

        for category, probability in zip(
            classes,
            probabilities
        ):

            st.write(
                f"**Kategori {category}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        st.divider()

        with st.expander("📋 View Patient Input"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# TAB 2 — DATASET
# ============================================================

with tab_dataset:

    st.markdown(
        '<div class="section-title">'
        'Indian Liver Patient Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Dataset yang digunakan dalam proyek ini adalah
    **Indian Liver Patient Dataset**.

    Dataset berisi beberapa karakteristik klinis pasien
    yang digunakan sebagai feature untuk melakukan
    klasifikasi.
    """)

    st.divider()

    st.subheader("Feature Description")

    feature_description = pd.DataFrame({

        "Feature": [
            "Age",
            "Gender",
            "Total_Bilirubin",
            "Direct_Bilirubin",
            "Alkaline_Phosphotase",
            "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase",
            "Total_Protiens",
            "Albumin",
            "Albumin_and_Globulin_Ratio"
        ],

        "Description": [
            "Age of the patient.",
            "Gender of the patient. 1 = Male, 0 = Female.",
            "Total bilirubin level.",
            "Direct bilirubin level.",
            "Alkaline phosphotase level.",
            "Alamine aminotransferase level.",
            "Aspartate aminotransferase level.",
            "Total protein level.",
            "Albumin level.",
            "Albumin and globulin ratio."
        ]

    })

    st.dataframe(
        feature_description,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Target Variable")

    st.write(
        "**Kategori** merupakan target/label yang "
        "digunakan dalam proses klasifikasi."
    )

    st.markdown("""
    - **Kategori 1**
    - **Kategori 2**
    """)


# ============================================================
# TAB 3 — MODEL
# ============================================================

with tab_model:

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Model yang digunakan dalam aplikasi ini adalah
    **Logistic Regression**.

    Pemilihan model dilakukan melalui perbandingan
    beberapa algoritma machine learning pada tahap eksperimen.
    """)

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    with col2:
        st.metric(
            "Accuracy",
            "71.05%"
        )

    with col3:
        st.metric(
            "Precision",
            "73.53%"
        )

    with col4:
        st.metric(
            "F1-Score",
            "81.97%"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "SVM",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "Gaussian Naive Bayes"
        ],

        "Accuracy (%)": [
            71.05,
            71.05,
            66.67,
            64.04,
            62.28,
            52.63
        ]

    })

    comparison = comparison.sort_values(
        by="Accuracy (%)",
        ascending=False
    )

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    st.subheader("Final Model Evaluation")

    evaluation = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],

        "Score (%)": [
            71.05,
            73.53,
            92.59,
            81.97
        ]

    })

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True
    )

    st.info("""
    Accuracy, precision, recall, dan F1-score digunakan
    untuk memberikan gambaran performa model dari beberapa
    perspektif evaluasi.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🩺 Indian Liver Patient Classification '
    '| Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)

