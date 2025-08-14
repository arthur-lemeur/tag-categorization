import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from bs4 import BeautifulSoup
import re
from bs4 import MarkupResemblesLocatorWarning
import warnings
import numpy as np
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import unicodedata
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import json

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Launch: 
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

########
# DEV

# import time
# import webbrowser
# import threading
# import requests

# nltk.download('punkt', download_dir='./nltk_data')
# nltk.download('stopwords', download_dir='./nltk_data')
# nltk.download('punkt_tab', download_dir='./nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data')
# nltk.download('averaged_perceptron_tagger_eng', download_dir='./nltk_data')
# nltk.download('wordnet', download_dir='./nltk_data')

# run_id = "e8376fe71c9e4e468f75ab4c0821bb74"
# model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
# joblib.dump(model, 'model.pkl')

# client = mlflow.tracking.MlflowClient()
# run = client.get_run(run_id)

# infos = {
#     "model_type": "Logistic Regression",
#     "run_id": run_id,
#     "performance": run.data.metrics,
#     "hyperparameters": run.data.params,
#     "tags": run.data.tags
# }

# with open("model_infos.json", "w") as f:
#     json.dump(infos, f)


# # Fonction pour ouvrir le navigateur automatiquement
# def open_browser():
#     time.sleep(1.5)  # Attendre que le serveur soit prêt
#     webbrowser.open('http://localhost:8000')

# # Au démarrage de l'application
# @app.on_event("startup")
# async def startup_event():
#     # Lancer l'ouverture du navigateur dans un thread séparé
#     threading.Thread(target=open_browser, daemon=True).start()

########

def initialize_nltk():
    nltk_data_path = os.path.join(os.path.dirname(__file__), 'data/nltk_data')
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

initialize_nltk()

class PredictionInput(BaseModel):
    Title: str
    Body: str

class PredictionOutput(BaseModel):
    predicted_labels: list[str]
    prediction_binary: Optional[list[int]] = None
    probability: Optional[list[float]] = None
    all_predictions: object

class TagProbabilityInput(BaseModel):
    Title: str
    Body: str
    tag: str

client = MlflowClient()
most_used_tags_df = pd.read_csv("data/dataframes/most_used_tags_20.csv", encoding='utf-8')
TAGS_LIST = most_used_tags_df["Word"].tolist()

app = FastAPI(title="Tag Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost:3000"] pour un frontend spécifique
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route pour servir votre page HTML principale
@app.get("/")
async def read_index():
    return FileResponse('static/form.html')  # Remplacez par le nom de votre fichier

# Charger le modèle ET le vectorizer
def load_model_and_vectorizer():
    try:
        # Vérifier si les fichiers existent
        model_path = 'model.pkl'  # ou le nom de votre fichier de modèle
        vectorizer_path = 'tfidf_vectorizer_corrected.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer non trouvé: {vectorizer_path}")
        
        # Charger les fichiers
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("Modèle et vectorizer chargés avec succès")
        return model, vectorizer
        
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de chargement: {str(e)}")

# Charger seulement si les fichiers existent
model, vectorizer = None, None

@app.on_event("startup")
async def startup_event():
    global model, vectorizer
    try:
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        print(f"Modèles non disponibles: {e}")
        # L'app démarre quand même, mais sans ML

# Route de test pour vérifier le statut
@app.get("/api/status")
async def get_status():
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "status": "ready" if (model and vectorizer) else "models_missing"
    }

lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))
wordnet_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}

def clean_text(text):
    # Étape 1: Nettoyer le HTML
    cleaned_text = BeautifulSoup(text, "lxml").get_text()
    
    # Étape 2: Normaliser les caractères Unicode (résout beaucoup de problèmes d'encodage)
    cleaned_text = unicodedata.normalize('NFKD', cleaned_text)
    
    # Étape 3: Gérer les retours à la ligne et espaces multiples
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remplace tous les espaces multiples par un seul
    
    # Étape 4: Gérer les apostrophes problématiques
    # Remplacer les apostrophes typographiques par des apostrophes standard
    cleaned_text = re.sub(r"[’‘'`´]", "", cleaned_text)
    
    # Gérer les contractions courantes AVANT la conversion en minuscules
    contractions = {
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bn't\b": " not",
        r"\b're\b": " are",
        r"\b've\b": " have",
        r"\b'll\b": " will",
        r"\b'd\b": " would",
        r"\bI'm\b": "I am",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is"
    }
    
    for contraction, replacement in contractions.items():
        cleaned_text = re.sub(contraction, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # Étape 5: Conversion en minuscules
    cleaned_text = cleaned_text.lower()
    
    # Étape 6: Préserver les termes techniques spéciaux
    cleaned_text = cleaned_text.replace('c#', 'csharpxxx').replace('c #', 'csharpxxx')
    cleaned_text = cleaned_text.replace('c++', 'cplusxxx').replace('c ++', 'cplusxxx')
    
    # Étape 7: Supprimer les caractères non-alphabétiques (URLs, chiffres, ponctuation)
    # Mais garder les espaces pour la tokenisation
    cleaned_text = re.sub(r"http\S+|www\.\S+", ' ', cleaned_text)  # URLs
    cleaned_text = re.sub(r"[^a-zA-Z\s]", ' ', cleaned_text)  # Tout sauf lettres et espaces
    
    # Étape 8: Nettoyer les espaces multiples finaux
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def lemma_fct(list_words):
    # Filtrer les mots vides avant le POS tagging pour plus d'efficacité
    words_filtered = [word for word in list_words if word and len(word) > 1]
    
    if not words_filtered:
        return []
    
    pos_tags = nltk.pos_tag(words_filtered)
    lem_w = [lemmatizer.lemmatize(word, pos=wordnet_map.get(pos[0].upper(), wordnet.NOUN)) 
             for word, pos in pos_tags]
    return lem_w

def c_sharp_pp(words_list):
    """Restaure les termes techniques spéciaux"""
    return [
        word.replace("sharpxxx", "#").replace("plusxxx", "++")
        if "sharpxxx" in word or "plusxxx" in word else word
        for word in words_list
    ]

def process_text(text, remove_stopwords=True):
    """
    Pipeline complet de traitement de texte
    
    Args:
        text (str): Texte à nettoyer
        remove_stopwords (bool): Supprimer les mots vides ou non
    
    Returns:
        str: Texte nettoyé et lemmatisé
    """
    # Étape 1: Nettoyage initial
    cleaned_text = clean_text(text)
    
    # Étape 2: Tokenisation
    tokens = word_tokenize(cleaned_text)
    
    # Étape 3: Filtrage des mots vides (optionnel)
    if remove_stopwords:
        tokens = [word for word in tokens if word.lower() not in sw and len(word) > 1]
    
    # Étape 4: Restauration des termes techniques
    tokens = c_sharp_pp(tokens)
    
    # Étape 5: Lemmatisation
    tokens = lemma_fct(tokens)
    
    # Étape 6: Filtrage final (enlever les mots trop courts ou vides)
    tokens = [word for word in tokens if word and len(word) > 1]
    
    return ' '.join(tokens)

@app.get("/")
def read_root():
    return {"message": "Tag Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Votre preprocessing existant...
        data = pd.DataFrame([input_data.dict()])
        data["Title"] = data["Title"].apply(lambda x: process_text(clean_text(x), remove_stopwords=True))
        data["Body"] = data["Body"].apply(lambda x: process_text(clean_text(x), remove_stopwords=True))
        data['Merged_doc'] = data.apply(lambda x: x['Title'] + ' ' + x['Body'], axis=1)
        
        # Vectorisation
        text_vectorized = vectorizer.transform(data['Merged_doc'])
        
        # Prédiction
        prediction_binary = model.predict(text_vectorized)[0]
        
        # Convertir les prédictions binaires en noms de classes
        predicted_labels = [
            TAGS_LIST[i] for i, pred in enumerate(prediction_binary) if pred == 1
        ]
        
        # Probabilités
        probabilities = None
        predicted_labels_with_probs = predicted_labels
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(text_vectorized)[0].tolist()
                top3_indices = np.argsort(probabilities)[-3:][::-1]
                top3_tags = [TAGS_LIST[i] for i in top3_indices if i < len(TAGS_LIST)]
                top3_probs = [float(probabilities[i]) for i in top3_indices if i < len(TAGS_LIST)]
            except Exception:
                probabilities = None
                predicted_labels=predicted_labels_with_probs,
        
            return PredictionOutput(
                all_predictions = probabilities,
                predicted_labels = top3_tags,
                probability = top3_probs,
                # prediction_binary=prediction_binary.tolist(),
            )
        else:
            # Pas de probabilités disponibles
            return PredictionOutput(
                predicted_labels = predicted_labels_with_probs,
                probability = None
            )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/infos")
def get_complete_model_info():
    try:
        with open("model_infos.json", "r") as f:
            infos = json.load(f)
        return infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")