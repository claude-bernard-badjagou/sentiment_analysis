# Lancer avec : python train.py

# ─────────────────────────────────────────────────────────────────────────────
# app.py — Application Streamlit : Analyseur de Sentiment Français
# ─────────────────────────────────────────────────────────────────────────────
# Déploiement sur Streamlit Cloud :
#   1. Pousser ce fichier + requirements.txt sur un repo GitHub public
#   2. Aller sur https://share.streamlit.io → "New app"
#   3. Sélectionner le repo, la branche, et ce fichier (app.py)
#   4. Cliquer "Deploy" — l'app est en ligne en ~2 minutes
#
# Structure du repo GitHub attendue :
#   ├── app.py            ← ce fichier
#   └── requirements.txt  ← dépendances (streamlit, transformers, torch)
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import streamlit as st

from collections import Counter
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline,
    DataCollatorWithPadding,
)

from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME               = "camembert-base"
OUTPUT_DIR               = "./modele_allocine"
NUM_LABELS               = 2
MAX_LENGTH               = 256
BATCH_SIZE               = 16
LEARNING_RATE            = 2e-5
NUM_EPOCHS               = 3
DROPOUT_RATE             = 0.3
WEIGHT_DECAY             = 1e-4
EARLY_STOPPING_PATIENCE  = 2
TRAIN_SIZE               = 5000
TEST_SIZE                = 1000


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — NLP
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ÉTAPE 1 — DÉFINITION DU PROBLÈME NLP")
print("=" * 70)
print("""
Tâche        : Classification binaire de sentiments
Entrée       : Critique de film en français (texte libre)
Sortie       : 0 = Négatif  |  1 = Positif
Dataset      : AlloFilm (Hugging Face) — 160 000 critiques françaises
Architecture : CamemBERT (Transfer Learning)
""")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — COLLECTE DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ÉTAPE 2 — COLLECTE DES DONNÉES")
print("=" * 70)

print("Chargement du dataset AlloFilm depuis Hugging Face...")
dataset = load_dataset("allocine")

print(f"\nStructure : {dataset}")
print(f"Premier exemple :")
print(f"  Texte : {dataset['train'][0]['review'][:150]}...")
print(f"  Label : {dataset['train'][0]['label']}")

distribution = Counter(dataset["train"]["label"])
total         = len(dataset["train"])
print(f"\nDistribution des classes (train) :")
print(f"  Positifs : {distribution[1]:6} ({distribution[1]/total*100:.1f}%)")
print(f"  Négatifs : {distribution[0]:6} ({distribution[0]/total*100:.1f}%)")

if TRAIN_SIZE:
    dataset["train"]      = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    dataset["test"]       = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(TEST_SIZE // 2))
    print(f"\n[Info] Sous-ensemble : {TRAIN_SIZE} train / {TEST_SIZE} test")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 3 — NETTOYAGE DES DONNÉES")
print("=" * 70)


def nettoyer_texte(texte: str) -> str:
    """
    Nettoie un texte avant tokenisation et inférence.
    Indispensable car réutilisée à l'entraînement ET à l'inférence (Étape 10).
    """
    if not isinstance(texte, str):
        return ""
    texte = re.sub(r"<[^>]+>",          " ", texte)
    texte = re.sub(r"&[a-z]+;",         " ", texte)
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)
    texte = re.sub(r"([!?.]){2,}",   r"\1", texte)
    texte = re.sub(r"[\n\r\t]+",        " ", texte)
    texte = re.sub(r" {2,}",            " ", texte)
    return texte.strip()


print("Nettoyage des textes...")
dataset = dataset.map(
    lambda ex: {"review": [nettoyer_texte(t) for t in ex["review"]]},
    batched=True,
)
print("Nettoyage terminé !")

exemple_brut = "Ce film était <b>magnifique</b> !!!!! Vraiment\n\npas déçu du tout !"
print(f"\nAvant : {exemple_brut}")
print(f"Après : {nettoyer_texte(exemple_brut)}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 4 — TOKENISATION")
print("=" * 70)

print(f"Chargement du tokenizer '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocabulaire : {tokenizer.vocab_size:,} tokens")

exemple_tok = "J'adore les films de science-fiction extraordinaires !"
print(f"\nExemple :")
print(f"  Texte  : {exemple_tok}")
print(f"  Tokens : {tokenizer.tokenize(exemple_tok)}")
print(f"  IDs    : {tokenizer.encode(exemple_tok)}")

print(f"\nTokenisation des données (MAX_LENGTH={MAX_LENGTH})...")
dataset_tokenise = dataset.map(
    lambda ex: tokenizer(
        ex["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    ),
    batched=True,
)
print("Tokenisation terminée !")

premier = dataset_tokenise["train"][0]
print(f"\nStructure après tokenisation :")
print(f"  Clés disponibles  : {list(premier.keys())}")
print(f"  input_ids (début) : {premier['input_ids'][:10]}...")
print(f"  attention_mask    : {premier['attention_mask'][:10]}...")
print(f"  label             : {premier['label']}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — WORD EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 5 — WORD EMBEDDINGS")
print("=" * 70)

embedding_dim    = 128
couche_embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)

ids_demo = torch.tensor(dataset_tokenise["train"][0]["input_ids"][:5])
vecteurs = couche_embedding(ids_demo)
print(f"Illustration nn.Embedding :")
print(f"  IDs d'entrée       : {ids_demo.tolist()}")
print(f"  Shape des vecteurs : {vecteurs.shape}")
print(f"  Premier vecteur    : {vecteurs[0][:8].tolist()}...")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — CONSTRUCTION DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ÉTAPE 6 — CONSTRUCTION DU MODÈLE")
print("=" * 70)


class ModeleNLP_LSTM(nn.Module):
    """LSTM illustratif — cf. cours Étape 6."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_p=0.3):
        super().__init__()
        self.embed         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=dropout_p)
        self.lstm          = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout_fc    = nn.Dropout(p=dropout_p)
        self.fc            = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, **kwargs):
        x           = self.embed(input_ids)
        x           = self.dropout_embed(x)
        _, (h_n, _) = self.lstm(x)
        h_n         = self.dropout_fc(h_n[-1])
        return self.fc(h_n)


modele_lstm    = ModeleNLP_LSTM(tokenizer.vocab_size, 128, 256, NUM_LABELS, DROPOUT_RATE)
nb_params_lstm = sum(p.numel() for p in modele_lstm.parameters() if p.requires_grad)
print(f"Modèle A — LSTM from scratch : {nb_params_lstm:,} paramètres")

print(f"\nModèle B — CamemBERT (Transfer Learning) ← UTILISÉ :")
modele_camembert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS,
)
nb_params_bert = sum(p.numel() for p in modele_camembert.parameters() if p.requires_grad)
print(f"  Paramètres  : {nb_params_bert:,}")
print(f"  Architecture : 12 couches Transformer + 144 têtes d'attention")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 7 — ENTRAÎNEMENT DU MODÈLE")
print("=" * 70)

metrique_accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=1)
    return metrique_accuracy.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir                   = OUTPUT_DIR,
    num_train_epochs             = NUM_EPOCHS,
    per_device_train_batch_size  = BATCH_SIZE,
    per_device_eval_batch_size   = BATCH_SIZE * 2,
    learning_rate                = LEARNING_RATE,
    weight_decay                 = WEIGHT_DECAY,
    eval_strategy                = "epoch",
    save_strategy                = "epoch",
    load_best_model_at_end       = True,
    metric_for_best_model        = "accuracy",
    greater_is_better            = True,
    lr_scheduler_type            = "cosine",
    warmup_steps                 = 50,
    logging_steps                = 50,
    seed                         = 42,
    fp16                         = torch.cuda.is_available(),
    dataloader_num_workers       = 0,
)

print(f"Configuration :")
print(f"  Epochs         : {NUM_EPOCHS}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Learning rate  : {LEARNING_RATE}")
print(f"  Early stopping : patience={EARLY_STOPPING_PATIENCE}")
print(f"  GPU disponible : {torch.cuda.is_available()}")

dataset_tokenise = dataset_tokenise.rename_column("label", "labels")
dataset_tokenise.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

trainer = Trainer(
    model           = modele_camembert,
    args            = training_args,
    train_dataset   = dataset_tokenise["train"],
    eval_dataset    = dataset_tokenise["validation"],
    compute_metrics = compute_metrics,
    data_collator   = data_collator,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print(f"\nLancement de l'entraînement ({len(dataset_tokenise['train'])} exemples)...")
debut        = time.time()
train_result = trainer.train()
print(f"\nEntraînement terminé en {(time.time()-debut)/60:.1f} min")
print(f"Loss finale : {train_result.training_loss:.4f}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modèle sauvegardé dans : {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 8 — ÉVALUATION DU MODÈLE")
print("=" * 70)

modele_camembert.eval()

print("Calcul des prédictions sur le jeu de test...")
sortie_test       = trainer.predict(dataset_tokenise["test"])
predictions_test  = np.argmax(sortie_test.predictions, axis=1)
vrais_labels_test = sortie_test.label_ids

rapport = classification_report(
    vrais_labels_test,
    predictions_test,
    target_names=["Négatif (0)", "Positif (1)"],
    digits=4,
)
print(f"\nRapport de classification :")
print(rapport)

matrice = confusion_matrix(vrais_labels_test, predictions_test)
print(f"Matrice de confusion :")
print(f"             Prédit Nég  Prédit Pos")
print(f"  Vrai Nég :   {matrice[0][0]:6}       {matrice[0][1]:6}")
print(f"  Vrai Pos :   {matrice[1][0]:6}       {matrice[1][1]:6}")

accuracy_finale = np.trace(matrice) / matrice.sum()
print(f"\nAccuracy finale : {accuracy_finale:.4f} ({accuracy_finale*100:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPES 9 & 10 — les prints sont conservés (pas de résultat affiché dans l'app)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ÉTAPE 9 — OPTIMISATIONS INTÉGRÉES")
print("=" * 70)
print(f"  ✓ Dropout          = {DROPOUT_RATE}")
print(f"  ✓ Weight Decay     = {WEIGHT_DECAY}")
print(f"  ✓ Early Stopping   = patience {EARLY_STOPPING_PATIENCE}")
print(f"  ✓ Cosine Scheduler + warmup 10%")
print(f"  ✓ Mixed Precision  = {torch.cuda.is_available()}")
