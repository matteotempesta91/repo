# Monitoraggio della reputazione online di un’azienda

## Descrizione del progetto

Questo progetto ha l’obiettivo di monitorare la reputazione online di un’azienda attraverso l’analisi automatica del sentiment dei contenuti pubblicati sui social media.

L’idea è utilizzare un modello di machine learning pre-addestrato per classificare i testi in sentiment:
- positivo
- neutro
- negativo

In questo modo è possibile osservare l’evoluzione della percezione degli utenti nel tempo e supportare decisioni rapide e basate sui dati.

## Obiettivi

Gli obiettivi principali del progetto sono due:

- monitorare la bontà del modello di sentiment analysis nel tempo, verificando che le sue prestazioni rimangano affidabili
- monitorare il sentiment degli utenti nei confronti dell’azienda, per osservare l’andamento della reputazione online
- costruire una pipeline CI/CD per validare il progetto


In questo modo il progetto unisce una componente di controllo della qualità del modello a una componente di business, legata alla percezione dell’azienda sui social media.

## Assunzioni

Per adattare il progetto al caso d’uso aziendale, sono state fatte le seguenti assunzioni di simulazione:

- tutte le review presenti nel dataset BanglaEcomReviewCorpus sono considerate come se fossero feedback rivolti all’azienda MLOps Innovators Inc.
- per monitorare la bontà del modello, è possibile confrontare le etichette originali del dataset con le predizioni prodotte dal modello utilizzato

Queste assunzioni permettono di simulare uno scenario realistico di monitoraggio del sentiment aziendale, pur utilizzando un dataset pubblico non raccolto direttamente sull’azienda.

## Dataset

Per questo progetto è stato utilizzato il dataset BanglaEcomReviewCorpus, disponibile a questo link:

- **Fonte:** https://data.mendeley.com/datasets/kkzfrvhbhp/2 
- **Descrizione:** Il dataset è cattura l’ampia gamma di sentimenti dei clienti riguardo alle esperienze di acquisto online su diverse piattaforme. Le fonti dei dati includevano recensioni dei clienti provenienti da siti web popolari come Daraz, Bikroy.com, Picabbo e altri. Il dataset contiene 8685 elementi etichettati, suddivisi in 3012 sentiment positivi (34,7%), 2881 sentiment negativi (33,2%) e 2792 sentiment neutri (32,1%), ottenendo così una distribuzione bilanciata che riflette la naturale varietà dei feedback del mondo reale.


## Modello utilizzato

Per l’analisi del sentiment è stato utilizzato il modello pre-addestrato:

- **Nome modello:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Link:** https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

Il modello classifica i testi in tre categorie:
- negativo
- neutro
- positivo

È stato scelto perché:
- è già addestrato per il sentiment analysis su testi simili a quelli social
- offre risultati affidabili su contenuti brevi e informali
- si adatta bene al caso d’uso del progetto

## Struttura del repository

La struttura principale del progetto è la seguente:

- `functions.py`: contiene le funzioni principali del progetto
- `inference.py`: gestisce l’inferenza del modello
- `tests/`: contiene i test automatici
- `tests/test_functions.py`: test delle funzioni principali
- `.github/workflows/ci.yml`: workflow di GitHub Actions per la pipeline CI
- `requirements.txt`: elenco delle dipendenze del progetto
- `README.md`: documentazione del progetto


## Esecuzione dell'inferenza e simulazione del monitoraggio

Il file `inference.py` consente di simulare il monitoraggio della bontà del modello nel tempo confrontando le etichette reali del dataset con le predizioni prodotte dal modello di sentiment analysis.

Il comportamento dello script è controllato da quattro variabili di input:

- `REVIEW_FOR_WEEK`: numero di review considerate per ogni settimana
- `CURRENT_WEEK`: numero di settimane utilizzate per calcolare la baseline iniziale a `t=0`
- `NUMBER_WEEK_TO_UPDATE`: numero di settimane aggiunte a ogni aggiornamento dell’inferenza
- `NUMBER_OF_UPDATE`: numero di aggiornamenti da eseguire

### Funzionamento

All’avvio dello script:
1. viene caricato il modello pre-addestrato
2. viene caricato il dataset `BanglaEcomReviewCorpus.xlsx`
3. le colonne vengono rinominate in modo coerente
4. vengono estratti i testi (`review`) e le etichette reali (`label`)
5. vengono richiesti in input i parametri per la simulazione
6. viene calcolata la baseline iniziale
7. vengono eseguiti gli aggiornamenti successivi, ricalcolando le performance del modello

### Esempio di esecuzione

Se si desidera analizzare:
- 100 review per settimana
- 4 settimane iniziali per la baseline
- aggiornamenti di 2 settimane per volta
- 3 aggiornamenti complessivi

si possono inserire i seguenti valori:
- Inserisci REVIEW_FOR_WEEK: 100
- Inserisci CURRENT_WEEK: 4
- Inserisci NUMBER_WEEK_TO_UPDATE: 2
- Inserisci NUMBER_OF_UPDATE: 3


## Output della funzione `display_performance`

La funzione `display_performance` ha il compito di visualizzare a schermo le metriche principali del modello e di salvare su file i grafici generati durante la simulazione.

### Output a schermo

Durante l’esecuzione, la funzione stampa:

- il sentiment score calcolato sui campioni analizzati
- l’accuracy score del modello

Queste informazioni permettono di avere un primo riscontro immediato sulle performance del modello.

### Output salvati su file

La funzione salva automaticamente i grafici nella cartella `plots/`, che viene creata se non esiste già.

I file generati sono:

- `plots/sentiment_plot_week_<week>_total_samples_<samples>.png`
- `plots/confusion_matrix_week_<week>_total_samples_<samples>.png`
- `plots/metrics_plot_week_<week>_total_samples_<samples>.png`

### Significato dei grafici

- **Sentiment plot**: mostra la distribuzione delle predizioni nei tre sentiment
- **Confusion matrix**: visualizza il confronto tra etichette reali e predette
- **Metrics plot**: riassume le metriche medie di precision, recall e f1-score

### Esempio

Se il numero di settimane analizzate è 4 e i campioni totali sono 400, i file salvati avranno nomi simili a:

- `plots/sentiment_plot_week_4_total_samples_400.png`
- `plots/confusion_matrix_week_4_total_samples_400.png`
- `plots/metrics_plot_week_4_total_samples_400.png`

## Pipeline CI/CD

Il progetto include una pipeline di integrazione continua configurata con GitHub Actions.

Il workflow si attiva:
- ad ogni push sul branch `main`
- ad ogni pull request

La pipeline esegue i seguenti passaggi:
1. checkout del repository
2. configurazione di Python
3. installazione delle dipendenze
4. esecuzione dei test automatici

I test servono a verificare che le principali funzioni del progetto si comportino correttamente e che le modifiche introdotte non compromettano il funzionamento del codice.

In particolare, i test validano:
- il corretto calcolo delle metriche di performance
- la coerenza dell’output generato dalle funzioni principali
- il funzionamento della logica di simulazione dell’inferenza

## Deploy

Il deploy su Hugging Face non è stato realizzato perché facoltativo. Il progetto si è concentrato sulla parte di analisi del sentiment, pipeline CI/CD e monitoraggio delle performance del modello.
