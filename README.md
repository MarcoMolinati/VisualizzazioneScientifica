# Visualizzazione Scientifica - Marco Molinati, 923530

# Ucraina Russia - Sentiment e TimeSeries Analysis

Lo scopo di questo progetto/approfondimento è quello di eseguire delle analisi su una tematica attuale, ovvero quella del conflitto tra Ucraina e Russia, tramite i tweet degli utenti nel periodo precedente e durante il conflitto, all’inizio dell’invasione russa.

La parte relativa alla raccolta, analisi ed elaborazione dei dati è stata interamente realizzata da me, l’unico processo automatico utilizzato è relativo all’assegnazione dei sentimenti nei tweet, in quanto è stata utilizzata una libreria per l’elaborazione del linguaggio naturale della NLTK (*Natural Language ToolKit*), la quale possiede delle pipeline già addestrate e testate per la cosiddetta ‘unsupervided learning’

Racchiudendo in alcuni punti i passaggi seguiti per l’analisi, possiamo distinguere: 

- Data Collection
- Data Cleaning
- TimeSeries Analysis
- Text Preprocessing
- Text Classification
- Jaccard Similarity
- K-Means Clustering

## Data Collection

In questa fase del progetto, tramite la libreria python Tweepy è stato possibile avere accesso ai metadati relativi ai tweet grazie alle API di Twitter. Una volta eseguita l’autenticazione tramite le chiavi personali segrete che si ottengono con un account developer, è stato possibile lanciare una query per l’estrazione dei tweets; il filtro di ricerca è stato relativo alla parola *Ukraine* e sono stati estratti i tweet nella fascia temporale da inizio Gennaio al 5 Marzo con diversi attributi utili per le analisi. 

Di seguito una query di esempio per estrarre i dati

```python
client = tweepy.Client(bearer_token='')

keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))

tweets = client.search_recent_tweets(query=keyword, max_results=noOfTweet)

tweet_list = []
for tweet in (tweets.data):
    tweet_info = {
        'created_at': tweet.created_at,
        'id' : tweet.id,
        'original': tweet.text,
        'source': tweet.source,
    }
    tweet_list.append(tweet_info)

tweet_list = pd.DataFrame(tweet_list)
```

## Data Cleaning

Una volta raccolti i dati e organizzati per comodità in un file csv è iniziata la fase di Data Cleaning, che consiste nel ripulire i dati contenuti nel file in modo che siano pronti all’uso e in un formato facilmente modificabile. Sono stati rimossi i tweet in lingue diverse dall’inglese, in quanto le pipeline utilizzate nell’elaborazione del testo sono strutturate per lingue specifiche, in questo caso è stata scelta per comodità *‘en’*; è stata riformattata la colonna relativa all’username tenendo il solo nominativo, in quanto conteneva molti dati superflui per l’analisi da effettuare (bio, twitter handle, user profile id…). L’ultimo passo è stato quello di rimuovere, tramite espressioni regolari, dei pattern di caratteri che non sono utili all’analisi.

```python
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(
            r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem
        )
    )
    return df
```

## TimeSeries Analysis

In questa fase delle analisi è stata sfruttata la componente temporale presente nella colonna "date" del dataset. L’obiettivo è quello di verificare l’andamento dei tweet in funzione del tempo e controllare se ci sono dei momenti in cui ci sono stati dei valori fuori scala, oppure se vi è una distribuzione uniforme nella pubblicazione dei tweet. Accompagnano le analisi dei grafici relativi all’andamento settimanale dei tweet e dei giorni con più attività sulla piattaforma digitale.

## Text Preprocessing

In questa parte è venuta in aiuto la libreria SpaCy, la quale è uno degli standard per l’eleborazione del linguaggio naturale; è stata dunque utilizzata per rimuovere segni di punteggiatura, le cosiddette *‘stopwords’*, ovvero parole che non sono rilevanti nell’analisi del testo (es. the, and, is…) e successivamente sono stati applicati i processi di lemmatizzazione e tokenizzazione che sono relativamente l’attività di raggruppare le parole simili e che hanno lo stesso significato, mentre la tokenizzazione riguarda la divisione di un testo o stringa in *token*, ovvero parti di frase o parole che servono da punti chiave.

```python
nlp = en_core_web_sm.load()
tokenizer = RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()

def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)
```

## Text Classification

Questa è la parte in cui si studia, tramite NLP (*Natural Language Processing*) il significato delle frasi del linguaggio naturale per cercare di cogliere delle informazioni, in questo caso i sentimenti dei tweet, che serviranno per fare una classificazione in funzione del tempo.

Per svolgere questa analisi sono state utilizzate due librerie, TextBlob e NLTK, le quali hanno delle pipeline già addestrate e consolidate per l’analisi dei sentimenti, in particolare sfruttano il database delle recensioni di IMDB per addestrare l’algoritmo.

Tramite TextBlob si calcolano la *polarità* e la *soggettività*; la polarità è un valore numerico che risiede nell’intervallo [-1, 1] dove i valori negativi indicano dei sentimenti negativi e quelli positivi l’opposto. La soggettività invece quantifica quanta opinione personale viene aggiunta al tweet.

Tramite NLTK, in particolare il modulo *vader*, si ottengono degli score associati ai tweet che riguardano le componenti positiva, negativa e neutrale, oltre che al *compound*, il quale è un valore numerico che, se pari a zero indica che il sentimento è neutrale, mentre se ≥ 0.05 è positivo e se ≤ -0.05 è negativo. In seguito a questa analisi è stato dunque ricostruito il dataset di partenza aggiungendo delle colonne relative allo score dei vari tweet e il sentimento associato

```python
tw_list[["polarity", "subjectivity"]] = tw_list["clean_tweet"].apply(
    lambda Text: pd.Series(TextBlob(Text).sentiment)
)
for index, row in tw_list["clean_tweet"].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]

    if neg > pos:
        tw_list.loc[index, "sentiment"] = "negative"
    elif pos > neg:
        tw_list.loc[index, "sentiment"] = "positive"
    else:
        tw_list.loc[index, "sentiment"] = "neutral"

    tw_list.loc[index, "neg"] = neg
    tw_list.loc[index, "neu"] = neu
    tw_list.loc[index, "pos"] = pos
    tw_list.loc[index, "compound"] = comp
```

## Jaccard Similarity

Per questa sezione e quella successiva di Clustering, sono stati definiti degli insiemi di parole relative a diversi temi, in particolare Economy, Social, Culture, Health.

L’obiettivo della Jaccard Similarity è trovare un valore numerico, uno score, utile a capire quanto due insiemi sono simili tra di loro; in questo caso gli insiemi sono i set di parole per tematica e i diversi tweet.

La Jaccard Similarity, o Jaccard Score si ottiene secondo la seguente formula: $jaccard(A, B) = \large \frac{|A \cap B|}{|A \cup B|}$ dove A, B sono i due insiemi considerati.

Di seguito le funzioni Python utilizzate per il calcolo dello score:

```python
def jaccard_similarity(group, tweet):
    group = set(group)
    try:
        tweet = set(tweet)
        nominator = group.intersection(tweet)
        denominator = group.union(tweet)
        similarity = len(nominator)/len(denominator)
        return similarity
    except:
        print(tweet)

def get_scores(group, tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores
```

Una volta ottenuto uno score per le diverse categorie, si può assegnare la classe corretta ai tweet a seconda del punteggio.

## Clustering

Anche in questa fase delle analisi sono tornati utili i quattro set definiti per le parole relative alle diverse tematiche, oltre che al dataset creato con la Jaccard Similarity contenente i tweet e il relativo set di appartenenza.

La suddivisione dei tweets nelle quattro categorie è abbastanza equa, in quanto le parole possono essere sovrapposte ed intese con più significati.

Per la parte di Clustering l’obiettivo è stato quello di utilizzare l’algoritmo *K-Means* per cercare di individuare dei cluster, o classi di appartenenza, tra due diversi attributi e per verificare se ci fosse un legame tra i dati. Le categorie scelte per l’analisi sono:

- Economic - Social
- Social - Culture
- Economic - Health
- Economic - Culture

Per individuare il numero di cluster da utilizzare per l’algoritmo si usa il cosiddetto ’Elbow Method’, in quanto visualizzando uno scatter plot per gli attributi non si vede ad occhio una netta divisione; esso consiste nel calcolare dei valori relativi alle distanze dei campioni dal centro dei clusters.

In questo caso per l’elbow method sono stati rappresentati i valori usando la Distorsione e l’Inerzia

> **Distorsione**: viene calcolata come la media delle distanze al quadrato dai centri dei cluster dei rispettivi cluster. Tipicamente, viene utilizzata la metrica della distanza euclidea.

> **Inerzia**: è la somma delle distanze al quadrato dei campioni dal centro del cluster più vicino.

Essendo la divisione nelle classi tematiche abbastanza equa, negli scatter plot si nota una relazione lineare tra gli attributi. Per una questione di performance sono stati presi per rappresentare i grafici i primi 5000 valori nell’array.
