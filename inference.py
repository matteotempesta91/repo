from transformers import pipeline
import pandas as pd
from functions import calculate_performance
from functions import display_performance

# Caricamento del modello di sentiment analysis
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
df = pd.read_excel("BanglaEcomReviewCorpus.xlsx", engine="openpyxl")
df = df.rename(columns={'English Translation': 'review', 'Sentiment':'label'})
df["label"] = df["label"].str.lower()
X = df.drop("label", axis=1)["review"].values
y = df["label"].values

# quante review del dataset BanglaEcomReviewCorpus vengono prese in considerazione per una settimana
REVIEW_FOR_WEEK = int(input("Inserisci REVIEW_FOR_WEEK: "))

# quante week vengono analizzate per calcolare la baseline attuale (a t=0)
CURRENT_WEEK = int(input("Inserisci CURRENT_WEEK: "))

# quante week vengono analizzate ad ogni update dell'inferenza
NUMBER_WEEK_TO_UPDATE = int(input("Inserisci NUMBER_WEEK_TO_UPDATE: "))

# quante volte viene aggiornato il modello (ogni update analizza NUMBER_WEEK_TO_UPDATE settimane)
NUMBER_OF_UPDATE = int(input("Inserisci NUMBER_OF_UPDATE: "))

# calcolo delle performance del modello per la baseline attuale (a t=0)
sentiment,model_metrics=calculate_performance(X,y,sentiment_task,REVIEW_FOR_WEEK,CURRENT_WEEK)
display_performance(sentiment,model_metrics,CURRENT_WEEK,REVIEW_FOR_WEEK*CURRENT_WEEK)

# calcolo delle performance del modello per ogni update (ogni update analizza NUMBER_WEEK_TO_UPDATE settimane)
for i in range(NUMBER_OF_UPDATE):
    CURRENT_WEEK += NUMBER_WEEK_TO_UPDATE
    sentiment,model_metrics=calculate_performance(X,y,sentiment_task,REVIEW_FOR_WEEK,CURRENT_WEEK)
    display_performance(sentiment,model_metrics,CURRENT_WEEK,REVIEW_FOR_WEEK*CURRENT_WEEK)