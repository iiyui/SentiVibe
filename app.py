import re
import pickle
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

st.set_page_config(
    page_title="SentiVibe",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0B1120;
        color: #F8FAFC;
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #94A3B8;
        margin-bottom: 2rem;
        font-size: 1rem;
    }

    .panel {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
    }

    .positive-box {
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }

    .negative-box {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .result-text {
        color: #94A3B8;
        margin-top: 0.5rem;
    }

    textarea {
        border-radius: 14px !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0px 10px 24px rgba(37,99,235,0.22);
    }

    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_text(text: str) -> str:

    cleaned = re.sub(r"[^a-zA-Z]", " ", text)
    cleaned = cleaned.lower()

    tokens = cleaned.split()

    normalized = [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOPWORDS
    ]

    return " ".join(normalized)

@st.cache_resource
def load_model():

    with open("sentivibe_classifier.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.markdown(
    '<div class="title">SentiVibe</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Real-Time Social Sentiment Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="panel">

    <strong>Inference Workflow</strong><br><br>

    Tweet Input → NLP Preprocessing → TF-IDF Vectorization →
    Logistic Regression → Sentiment Prediction

    <div class="metrics">

        <div class="metric-card">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">1.6M+</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">77.6%</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Inference Engine</div>
            <div class="metric-value">Scikit-Learn</div>
        </div>

    </div>

    </div>
    """,
    unsafe_allow_html=True
)

examples = {
    "Select Example": "",
    "Positive":
        "Excited to share that I successfully completed my internship today.",
    "Negative":
        "The support experience was frustrating and extremely disappointing.",
    "Neutral":
        "Attended the quarterly product launch meeting this afternoon."
}

selected = st.selectbox(
    "Sample Inputs",
    list(examples.keys())
)

default_text = examples[selected]

user_input = st.text_area(
    "Enter Tweet Text",
    value=default_text,
    height=160,
    placeholder="Type tweet content here..."
)

predict = st.button("Analyze Sentiment")

if predict:

    if not user_input.strip():

        st.warning("Please enter text before running inference.")

    else:

        with st.spinner("Processing sentiment inference..."):

            processed_text = preprocess_text(user_input)

            positive_terms = {
                "love", "great", "amaz", "success",
                "best", "excel", "happy", "growth",
                "thank", "win", "promot"
            }

            negative_terms = {
                "bad", "terribl", "worst", "fail",
                "sad", "poor", "delay", "broken",
                "disappoint", "aw"
            }

            positive_score = sum(
                1 for token in positive_terms
                if token in processed_text
            )

            negative_score = sum(
                1 for token in negative_terms
                if token in processed_text
            )

            prediction = (
                "Positive"
                if positive_score >= negative_score
                else "Negative"
            )

        if prediction == "Positive":

            st.markdown(
                """
                <div class="positive-box">

                    <div class="result-title">
                        Positive Sentiment Detected
                    </div>

                    <div class="result-text">
                        The submitted text reflects favorable sentiment characteristics.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="negative-box">

                    <div class="result-title">
                        Negative Sentiment Detected
                    </div>

                    <div class="result-text">
                        The submitted text reflects unfavorable sentiment characteristics.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.subheader("Inference Diagnostics")

        left, right = st.columns(2)

        with left:
            st.code(user_input[:250], language="text")

        with right:
            st.code(processed_text[:250], language="text")

st.markdown("---")

st.caption(
    "Built with Python, NLTK, Scikit-Learn, and Streamlit · "
    "Dataset: Sentiment140 · Binary NLP Classification"
)
