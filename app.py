import streamlit as st
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import nltk
from nltk.corpus import wordnet as wn
import tempfile

nltk.download('punkt')
nltk.download('wordnet')

# Load Whisper model and processor
AUDIO_MODEL = "openai/whisper-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL).to(device)
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    device=0 if torch.cuda.is_available() else -1
)

HESITATION_MARKERS = ["uh", "um", "erm", "ah", "eh"]

def compute_lexical_diversity(text):
    words = nltk.word_tokenize(text)
    return len(set(words)) / len(words) if words else 0

def count_hesitations(text):
    tokens = nltk.word_tokenize(text.lower())
    return sum(tokens.count(marker) for marker in HESITATION_MARKERS)

def compute_pitch_variability(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    return np.std(pitches) if len(pitches) > 0 else 0

def estimate_word_recall(transcript):
    tokens = nltk.word_tokenize(transcript)
    vague_terms = ["thing", "stuff", "it", "something"]
    vague_count = sum([transcript.lower().count(term) for term in vague_terms])
    return vague_count

def naming_task_score(transcript):
    expected = ["apple", "banana", "chair", "pen"]
    found = sum(1 for word in expected if word in transcript.lower())
    return found / len(expected)

def extract_features(file):
    y, sr = librosa.load(file)
    result = asr_pipeline(file)
    transcript = result['text'].strip()

    word_count = len(nltk.word_tokenize(transcript))
    sent_count = len(nltk.sent_tokenize(transcript))
    avg_words_per_sent = word_count / sent_count if sent_count else 0
    lexical_diversity = compute_lexical_diversity(transcript)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    pitch_variability = compute_pitch_variability(y, sr)
    hesitation_count = count_hesitations(transcript)
    recall_score = estimate_word_recall(transcript)
    naming_score = naming_task_score(transcript)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    if len(mfcc_mean) < 13:
        mfcc_mean = np.pad(mfcc_mean, (0, 13 - len(mfcc_mean)), mode='constant')

    features = [
        word_count, sent_count, avg_words_per_sent, lexical_diversity,
        tempo, zcr, energy, spectral_centroid,
        pitch_variability, hesitation_count, recall_score, naming_score
    ] + list(mfcc_mean)

    return features, transcript

# Streamlit UI
st.title("🧠 Cognitive Speech Analyzer")
uploaded_file = st.file_uploader("Upload an audio file (.wav only)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.audio(temp_path)
    st.info("Processing... This may take a few seconds.")
    features, transcript = extract_features(temp_path)

    if features:
        st.success("Transcription completed!")
        st.markdown(f"**Transcript:**\n\n{transcript}")

        st.markdown("### 🔬 Extracted Features")
        labels = [
            "Word Count", "Sentence Count", "Avg Words/Sentence", "Lexical Diversity",
            "Tempo", "ZCR", "Energy", "Spectral Centroid",
            "Pitch Variability", "Hesitations", "Recall Score", "Naming Score"
        ] + [f"MFCC-{i+1}" for i in range(13)]

        feature_dict = dict(zip(labels, features))
        for k, v in feature_dict.items():
            st.write(f"**{k}:** {v:.4f}")

        # Optional: visual clustering or anomaly detection if multiple samples are uploaded in future
    else:
        st.error("Failed to extract features from the audio.")