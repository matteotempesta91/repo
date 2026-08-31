from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def calculate_performance(X,y, sentiment_task,review_for_week, current_week):
  """
    Calcola le performance del modello su un sottoinsieme progressivo dei dati.

    La funzione:
    - seleziona un numero di review proporzionale a review_for_week e current_week
    - genera le predizioni tramite sentiment_task
    - calcola accuracy, confusion matrix e classification report
    - conta quante predizioni sono state assegnate a ciascun sentiment

    Args:
        X (list): Lista dei testi da analizzare.
        y (list): Lista delle etichette reali corrispondenti ai testi.
        sentiment_task (callable): Funzione o pipeline che restituisce la predizione del sentiment per un testo.
        review_for_week (int): Numero di review presenti per ogni settimana.
        current_week (int): Settimana corrente da analizzare.

    Returns:
        tuple:
            sentiment (dict): Dizionario con il conteggio delle predizioni per classe
                (negative, neutral, positive).
            model_metrics (dict): Dizionario con le metriche di valutazione:
                - accuracy_score
                - classification_report
                - confusion_matrix
  """
  number_review_to_analyze = review_for_week * current_week

  sentiment={
    "negative":None,
    "neutral":None,
    "positive":None,
  }

  model_metrics={
      "accuracy_score": None,
      "classification_report": None,
      "confusion_matrix":None
  }

  y_pred = [sentiment_task(text)[0]["label"] for text in X[:number_review_to_analyze]]

  labels=["negative","neutral","positive"]
  model_metrics["accuracy_score"] = accuracy_score(y[:number_review_to_analyze], y_pred[:number_review_to_analyze])
  model_metrics["confusion_matrix"] = confusion_matrix(y[:number_review_to_analyze], y_pred[:number_review_to_analyze], labels=labels)
  model_metrics["classification_report"] = classification_report(y[:number_review_to_analyze], y_pred[:number_review_to_analyze], labels=labels,output_dict=True)

  sentiment["negative"] = y_pred.count("negative")
  sentiment["neutral"] = y_pred.count("neutral")
  sentiment["positive"] = y_pred.count("positive")

  return sentiment, model_metrics


def update_current_week(new_current_week):
    """
    Aggiorna la variabile CURRENT_WEEK con un nuovo valore.

    Args:
        new_current_week (int): Il nuovo valore della settimana corrente.

    Returns:
        None
    """
    global CURRENT_WEEK
    CURRENT_WEEK = new_current_week


def display_performance(sentiment, model_metrics, number_week_analyzed, number_samples_analyzed):
    """
    Visualizza a schermo e salva su file le performance calcolate del modello.

    La funzione:
    - stampa il numero di settimane e campioni analizzati
    - stampa le metriche principali
    - salva un grafico a barre della distribuzione dei sentiment
    - salva la confusion matrix come heatmap
    - salva un grafico sintetico con precision, recall e f1-score medi

    Args:
        sentiment (dict): Dizionario con il conteggio delle predizioni per classe.
        model_metrics (dict): Dizionario con accuracy_score, classification_report e confusion_matrix.
        number_week_analyzed (int): Numero di settimane considerate.
        number_samples_analyzed (int): Numero di campioni analizzati.

    Returns:
        None
    """
    print("---------------")
    positive = sentiment["positive"]
    negative = sentiment["negative"]
    neutral = sentiment["neutral"]
    total = positive + negative + neutral

    if total > 0:
        sentiment_score = (positive - negative) / total
    else:
        sentiment_score = 0

    print("Sentiment score:")
    print(sentiment_score)
    print("Accuracy score:")
    print(model_metrics["accuracy_score"])
    print("---------------")

    os.makedirs("plots", exist_ok=True)

    # -------------------------
    # 1) Grafico sentiment
    # -------------------------
    sentiment_labels = list(sentiment.keys())
    sentiment_values = list(sentiment.values())

    filename_sentiment = f"plots/sentiment_plot_week_{number_week_analyzed}_total_samples_{number_samples_analyzed}.png"

    plt.figure(figsize=(7, 5))
    sns.barplot(x=sentiment_labels, y=sentiment_values, palette="viridis")
    plt.title("Distribuzione delle predizioni di sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Numero di predizioni")
    plt.tight_layout()
    plt.savefig(filename_sentiment)
    plt.close()

    # -------------------------
    # 2) Confusion matrix
    # -------------------------
    cm = model_metrics["confusion_matrix"]
    filename_cm = f"plots/confusion_matrix_week_{number_week_analyzed}_total_samples_{number_samples_analyzed}.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename_cm)
    plt.close()

    # -------------------------
    # 3) Metriche sintetiche medie
    # -------------------------
    report = model_metrics["classification_report"]
    class_labels = ["negative", "neutral", "positive"]

    precision_values = []
    recall_values = []
    f1_values = []

    for label in class_labels:
        precision_values.append(report[label]["precision"])
        recall_values.append(report[label]["recall"])
        f1_values.append(report[label]["f1-score"])

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    mean_f1 = np.mean(f1_values)

    metrics_names = ["Precision", "Recall", "F1-score"]
    metrics_values = [mean_precision, mean_recall, mean_f1]

    filename_metrics = f"plots/metrics_plot_week_{number_week_analyzed}_total_samples_{number_samples_analyzed}.png"

    plt.figure(figsize=(7, 5))
    sns.barplot(x=metrics_names, y=metrics_values, palette="magma")
    plt.ylim(0, 1)
    plt.title("Metriche medie del classification report")
    plt.ylabel("Valore medio")
    plt.xlabel("Metrica")
    plt.tight_layout()
    plt.savefig(filename_metrics)
    plt.close()




def update_performance(X,y,sentiment_task,review_for_week,starting_week,number_week_to_update):
  """
    Aggiorna le performance del modello a partire da una settimana iniziale.

    La funzione:
    - calcola la settimana corrente aggiornata
    - stima il numero di campioni da analizzare
    - calcola sentiment e metriche del modello
    - aggiorna la variabile globale CURRENT_WEEK

    Args:
        X (list): Lista dei testi da analizzare.
        y (list): Lista delle etichette reali corrispondenti.
        review_for_week (int): Numero di review presenti per ogni settimana.
        starting_week (int): Settimana di partenza.
        number_week_to_update (int): Numero di settimane da aggiungere alla starting_week.

    Returns:
        tuple:
            sentiment (dict): Conteggio delle predizioni per classe.
            model_metrics (dict): Metriche di valutazione del modello.
            number_samples_analyzed (int): Numero totale di campioni analizzati.
  """
  current_week = starting_week + number_week_to_update
  number_samples_analyzed=review_for_week*current_week
  sentiment,model_metrics = calculate_performance(X,y,sentiment_task,review_for_week,current_week)
  update_current_week(current_week)

  return sentiment,model_metrics,number_samples_analyzed