# 🧠 Cognitive Speech Analysis

This project is a speech intelligence pipeline for detecting early signs of cognitive impairment using audio samples. It transcribes speech, extracts linguistic and acoustic features, and applies unsupervised learning to cluster and detect anomalous patterns.

---

## 🚀 Features

- ✅ **Speech-to-Text** using Whisper (`openai/whisper-medium`)
- 📊 **Feature Extraction**:
  - Lexical diversity
  - Hesitation markers (`uh`, `um`, etc.)
  - Pitch variability
  - Word recall (vague terms like “thing”, “stuff”)
  - Naming task simulation
  - MFCCs, tempo, energy, ZCR, spectral centroid
- 📈 **Unsupervised ML**:
  - PCA for dimensionality reduction
  - K-Means clustering
  - Isolation Forest anomaly detection
- 📍 **Visual output** of PCA clusters and detected outliers

---

## 🗣️ Linguistic Features (from transcript)

| **Feature**              | **What it Measures**                                              | **Why It Matters** |
|--------------------------|--------------------------------------------------------------------|---------------------|
| **Word Count**           | Total number of words spoken                                      | Indicates verbosity and fluency |
| **Sentence Count**       | Total number of sentences                                         | Reflects structure and coherence |
| **Avg Words/Sentence**   | Mean number of words per sentence                                | Short or incomplete sentences may signal impairment |
| **Lexical Diversity**    | `unique_words / total_words`                                     | Lower diversity may reflect reduced vocabulary access |
| **Hesitation Count**     | Number of fillers (uh, um, ah, etc.)                             | Frequent pauses suggest retrieval problems or stress |
| **Word Recall Issues**   | Count of vague words (thing, stuff, it, something)               | Non-specific language hints at memory/language difficulty |
| **Naming Task Score**    | Fraction of expected target words recalled (e.g., “apple”)       | Used in clinical memory/language screening |

---

## 🔊 Acoustic Features (from audio signal)

| **Feature**              | **What it Measures**                                              | **Why It Matters** |
|--------------------------|--------------------------------------------------------------------|---------------------|
| **Tempo**                | Rate of speech (beats per minute)                                 | Slower tempo = cognitive or speech motor slowing |
| **Zero Crossing Rate (ZCR)** | Frequency of signal sign changes                             | Higher values = more noise or fricatives |
| **Energy**               | Average loudness level (RMS)                                      | Lower energy may indicate fatigue or hesitance |
| **Spectral Centroid**    | Brightness of the sound (where frequency energy is concentrated)  | Lower values = dull, muffled speech |
| **Pitch Variability**    | Variation in voice pitch using `piptrack()`                       | Flat pitch = monotonic voice, common in cognitive/affective disorders |
| **MFCCs (13 Coefficients)** | Mel Frequency Cepstral Coefficients capture vocal timbre      | Standard in speech recognition to model phonetic content |
"""
---
## 📦 Requirements

Install all dependencies using pip:

```bash
pip install torch transformers librosa scikit-learn matplotlib nltk
```
--- 
## ▶️ Usage

1. Place your `.wav` audio files inside the `samples/` directory.

2. Run the  Jupyter notebook:

3. The system will:

- Transcribe each file
- Extract linguistic and acoustic features
- Apply PCA, KMeans, and Isolation Forest
- Display a 2D plot of feature clusters and anomalies

---

### 📊 Output
- Transcripts of each file
- Console log of extracted features
- 2D plot of PCA components with clusters and outliers



