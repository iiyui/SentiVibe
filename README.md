# SentiVibe
Enterprise-Scale Sentiment Intelligence Pipeline for Social Media Analytics

## Author

## **iyui** [linkedin.com/in/iyui] 

Machine Learning • NLP • AI Systems Engineering

TweetPulse is a machine learning–driven Natural Language Processing (NLP) system engineered to perform high-volume sentiment classification on social media text. The platform processes raw tweet data, applies optimized text preprocessing and feature engineering pipelines, and predicts sentiment polarity using supervised learning techniques.

Built on a dataset of 1.6 million labeled tweets, the system delivers efficient and scalable sentiment inference with strong generalization performance.

---

## Overview

The project is designed to transform unstructured social media conversations into actionable sentiment insights.

Given a tweet as input, TweetPulse performs:

- Text normalization
- Linguistic preprocessing
- Feature vector generation
- Statistical sentiment classification

The final output predicts whether the input sentiment is:

- Positive
- Negative

The architecture emphasizes lightweight deployment, scalable inference, and production-oriented NLP workflows.

---

## Key Features

- Large-scale sentiment classification pipeline
- TF-IDF–based textual feature extraction
- Optimized NLP preprocessing workflow
- Logistic Regression inference engine
- Interactive Streamlit deployment interface
- Modular and reproducible training pipeline
- Lightweight deployment with low inference latency

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP Processing | NLTK |
| Machine Learning | Scikit-learn |
| Feature Engineering | TF-IDF Vectorization |
| Model Architecture | Logistic Regression |
| Training Environment | Google Colab |
| Deployment Interface | Streamlit |

---

## Repository Structure

```text
TweetPulse/
│
├── Twitter_Sentiment_Analysis_using_ML.ipynb
├── trained_model.sav
├── app.py
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `Twitter_Sentiment_Analysis_using_ML.ipynb` | Complete experimentation, preprocessing, training, and evaluation workflow |
| `trained_model.sav` | Serialized trained sentiment classification model |
| `app.py` | Streamlit-based web application for inference |
| `requirements.txt` | Dependency and runtime package definitions |

---

## System Architecture

The sentiment analysis pipeline follows a structured NLP workflow:

1. Raw tweet ingestion
2. Text cleaning and normalization
3. Lowercase transformation
4. Stopword removal
5. Word stemming
6. TF-IDF feature vector generation
7. Logistic Regression inference
8. Sentiment prediction output

---

## Dataset

### Sentiment140 Dataset

| Attribute | Value |
|---|---|
| Source | Kaggle |
| Dataset Size | 1,600,000 Tweets |
| Classification Type | Binary Sentiment Classification |

### Label Distribution

| Label | Sentiment |
|---|---|
| `0` | Negative |
| `1` | Positive |

The dataset contains large-scale real-world Twitter conversations, enabling robust supervised learning across diverse linguistic patterns.

---

## Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | 79.0% |
| Test Accuracy | 77.6% |

The model demonstrates strong generalization performance while maintaining computational efficiency suitable for lightweight production deployment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/TweetPulse.git
cd TweetPulse
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application locally:

```bash
streamlit run app.py
```

The application will automatically open in the browser and provide a real-time sentiment inference interface.

---

## Reproducing the Training Pipeline

To retrain or experiment with the model:

1. Open Google Colab
2. Upload the notebook file:
   `Twitter_Sentiment_Analysis_using_ML.ipynb`
3. Execute the notebook cells sequentially

---

## Future Enhancements

- Multi-class sentiment classification
- Transformer-based NLP architectures
- Real-time Twitter API integration
- Dockerized deployment pipeline
- Cloud-native inference serving
- Advanced analytics dashboard
- Multilingual sentiment processing

---

## License

This project is intended for educational, research, and portfolio purposes.

---

