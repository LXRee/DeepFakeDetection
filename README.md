# Istruzioni all'uso
Questa repository è nata in seno alla DeepFake challenge di Kaggle.

## Intuizioni 
La principale che ho avuto e che ho applicato è basata sul fatto che, come per noi umani, è probabilmente più semplice
anche per una macchina capire se un video è falso principalmente se è in movimento.

La secondaria è che sicuramente tanto vale usare lavoro già fatto. Per questo motivo si vedano le sezioni di cui parlerò
più tardi.

## Passi
1. **scaricare il dataset**. 
Il dataset contiene TOT video mp4. I video possono avere o faccia, o audio, o entrambi falsi.
E' presente anche un file `metadata.json` che contiene il nome del file e la sua label. La label è binaria, vero/falso.
Quindi non è dato sapere quali tra video e audio sono fake.
2. **Manipolazione del video**. 
Ogni video viene aperto, si seleziona un frame ogni 6 e di questo frame viene individuata
la faccia (con `MTCNN` e pesi pre-allenati). Per ogni faccia vengono estratte le feature con una `InceptionResNet` e pesi
pre-allenati. In questo caso ho modificato l'architettura della rete in modo tale da memorizzare le feature del penultimo 
layer FC, cosi' da avere piu' espressivita'. Si crea un `pandas` `DataFrame` all'interno del quale compaiono le colonne
`filename` (path al file), `video_embedding` per le features delle facce, `label` per la label corrispettiva.
Si vedano i file `DetectionPipeline.py` e `create_video_embeddings.py` nel particolare.
3. **Manipolazione dell'audio**.
Da ogni video viene estratta la corrispondente traccia audio (file `extract_audio.py`). Per ogni traccia video ne viene
realizzato una sorta di istrogramma, che viene analizzato da una rete convoluzionale pre-allenata. Anche di questo
vengono estratte le feature (file `create_audio_embedding.py`). Anche queste informazioni vengono spedite in un `pandas`
`dataframe`, con le colonne `filename`, col nome del file, e `audio_embedding` con le features dell'audio.
4. **Unione di audio e video**
Vengono uniti i due `DataFrame` in un unico, con quattro colonne: `filename`, `audio_embedding`, `video_embedding`, `label`.
Si veda il file `merge_embeddings`.
5. **Rete**
La rete da me costruita quindi prende in pasto un numero `RANDOM_CROP` di frame che vengono passati da una `LSTM`.
Le feature estratte dall'ultimo timestep di quest'ultima vengono concatenate in un unico layer FC assieme a quelle 
dell'audio. Da qui vanno all'output (di dimensione 1). Si veda il file `network.py`.
6. **Allenamento**
La rete viene allenata con normale backpropagation. La loss è la `BCEWithLogitsLoss`, che unisce la binary crossentropy
all'operazione `sigmoid` sull'input, che garantisce una maggiore stabilità numerica. Questa misura equivale alla log-loss
richiesta dalla challenge. Si vedano i file `model.py`, `training.py` e tutti gli altri associati ad essi.
7. **Test**
Il test viene eseguito con la stessa procedura di cui sopra.
8. **Submission**
Il codice per la submission ancora non è pronto, ma basta guardare uno dei qualsiasi notebook presenti in Kaggle per 
capire come farlo.

# Da fare
## Preparare codice per submission
## Modificare il codice di inferenza perché l'estrazione di feature da video e audio + inferenza siano operazioni continue
## VINCERE

