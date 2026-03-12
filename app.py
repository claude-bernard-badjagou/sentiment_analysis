import re
import torch
import streamlit as st
from transformers import pipeline

# ── DOIT ÊTRE EN PREMIER ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyseur de Sentiment 🎬",
    page_icon="🎬",
    layout="centered",
)

MAX_LENGTH = 256

# ── Nettoyage du texte ────────────────────────────────────────────────────────
def nettoyer_texte(texte: str) -> str:
    if not isinstance(texte, str):
        return ""
    texte = re.sub(r"<[^>]+>",          " ", texte)
    texte = re.sub(r"&[a-z]+;",         " ", texte)
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)
    texte = re.sub(r"([!?.]){2,}",   r"\1", texte)
    texte = re.sub(r"[\n\r\t]+",        " ", texte)
    texte = re.sub(r" {2,}",            " ", texte)
    return texte.strip()

# ── Chargement du modèle depuis HuggingFace Hub ───────────────────────────────
# @st.cache_resource : chargé une seule fois, réutilisé pour toutes les requêtes
@st.cache_resource
def charger_pipeline():
    return pipeline(
        "sentiment-analysis",
        # Modèle multilingue public — fonctionne directement sans entraînement
        # Une fois ton modèle pushé sur HF Hub, remplace par :
        # model = "ton-username/camembert-allocine"
        model     = "nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer = "nlptown/bert-base-multilingual-uncased-sentiment",
        device    = -1,  # CPU — Streamlit Cloud n'a pas de GPU
    )

# ── Mapping labels → sentiment lisible ───────────────────────────────────────
# nlptown retourne "1 star" à "5 stars" → on regroupe en 3 sentiments
def interpreter_label(label: str) -> tuple:
    nb_etoiles = int(label[0])  # "3 stars" → 3
    if nb_etoiles <= 2:
        return "NÉGATIF", "😢", "red"
    elif nb_etoiles == 3:
        return "NEUTRE", "😐", "orange"
    else:
        return "POSITIF", "😊", "green"

# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎬 Analyseur de Sentiment — Critiques de Films")
st.markdown("Entrez une critique en **français** et le modèle prédit le sentiment.")
st.divider()

texte_utilisateur = st.text_area(
    label       = "Votre critique :",
    placeholder = "Ex : Ce film était absolument magnifique, je recommande vivement !",
    height      = 150,
)

if st.button("Analyser", type="primary", use_container_width=True):

    if not texte_utilisateur.strip():
        st.warning("Veuillez saisir une critique avant d'analyser.")
    else:
        analyseur = charger_pipeline()

        with st.spinner("Analyse en cours..."):
            res = analyseur(
                nettoyer_texte(texte_utilisateur),
                truncation=True,
                max_length=MAX_LENGTH,
            )[0]

        label, emoji, couleur = interpreter_label(res["label"])
        confiance             = res["score"]

        st.divider()
        st.markdown(f"### Résultat : :{couleur}[{emoji} {label}]")
        st.metric(label="Confiance du modèle", value=f"{confiance*100:.1f}%")
        st.progress(confiance)

        with st.expander("Détails"):
            st.write(f"**Texte nettoyé :** {nettoyer_texte(texte_utilisateur)}")
            st.write(f"**Label brut :** `{res['label']}`")
            st.write(f"**Score :** `{confiance:.4f}`")

st.divider()

# ── Exemples cliquables ───────────────────────────────────────────────────────
st.markdown("#### Exemples à tester")
col1, col2 = st.columns(2)

with col1:
    st.markdown("😊 **Positifs**")
    st.code("Ce film est absolument magnifique !", language=None)
    st.code("Je recommande vivement ce chef-d'œuvre.", language=None)

with col2:
    st.markdown("😢 **Négatifs**")
    st.code("Quelle déception ! Scénario nul, deux heures perdues.", language=None)
    st.code("Le pire film de ma vie. Ennuyeux du début à la fin.", language=None)

st.caption("Modèle : nlptown/bert-base-multilingual-uncased-sentiment • Streamlit Cloud")
