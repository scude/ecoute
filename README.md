# Écoute — Pipeline audio RTSP → VAD → transcription (Whisper) → UI

Ce projet capture un flux audio (RTSP), découpe automatiquement la parole avec VAD, transcrit les segments avec Whisper, puis expose les résultats dans une interface Streamlit.

---

## 1) Fonctionnement global

Le pipeline suit cet ordre :

1. **Capture RTSP** (`capture_rtsp_audio.sh`)  
   Génère des fichiers WAV dans `audios/`.
2. **VAD** (`vad_segment.py`)  
   Détecte les portions parlées, exporte les segments dans `speech_segments/`, archive les WAV source traités dans `audios_processed/`.
3. **Transcription incrémentale** (`transcribe_segments.py`)  
   Lit les segments non traités, écrit les résultats dans SQLite, export JSON optionnel.
4. **UI Streamlit** (`app.py`)  
   Recherche/lecture des transcriptions.

Le script **`pipeline_runner.py`** orchestre les étapes 2 et 3 en mode one-shot ou boucle continue.

---

## 2) Installation

### Dépendances Python

```bash
pip install torch torchaudio silero-vad faster-whisper streamlit pandas pyyaml
```

### (Optionnel) RNNoise

```bash
sudo apt update
sudo apt install -y git autoconf automake libtool build-essential pkg-config
cd ~
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make
```

---

## 3) Démarrage rapide

1. Configurer le flux RTSP :

```bash
cp .env.sample .env
# éditer .env
```

2. Lancer la capture audio continue :

```bash
bash capture_rtsp_audio.sh
```

3. Lancer un run de pipeline (VAD + transcription) :

```bash
python pipeline_runner.py --once
```

4. Lancer l’UI :

```bash
streamlit run app.py
```

---

## 4) Orchestrateur `pipeline_runner.py`

Deux modes sont disponibles :

- `--once` : exécute **une seule** séquence `VAD -> transcription`, puis quitte.
- `--loop` : exécute la séquence en continu avec un intervalle configurable.

Exemples :

```bash
python pipeline_runner.py --once
python pipeline_runner.py --loop
```

### Robustesse incluse

- **Lockfile**: `/tmp/ecoute_pipeline.lock` (évite les exécutions concurrentes).
- **Logs JSON structurés**: niveau, message, durée par étape, compteurs de fichiers.
- **Backoff exponentiel** en cas d’erreur (jusqu’à `pipeline.max_backoff_seconds`).

---

## 5) Configuration

La configuration est centralisée dans **`config.yaml`**.  
Les variables d’environnement peuvent surcharger les valeurs du fichier.

### 5.1 Fichier `config.yaml`

Sections principales :

- `paths` : chemins d’entrée/sortie
- `pipeline` : intervalle et backoff
- `whisper` : modèle et paramètres d’inférence
- `vad` : paramètres de détection de parole

### 5.2 Variables d’environnement supportées

> Priorité: **variables d’environnement > config.yaml > valeurs par défaut internes**

#### Pipeline

- `PIPELINE_INTERVAL_SECONDS`  
  Intervalle entre deux runs en mode `--loop`.
- `PIPELINE_MAX_BACKOFF_SECONDS`  
  Durée maximale du backoff après erreur.

#### Chemins

- `INPUT_AUDIO_DIR`  
  Dossier des WAV de capture (ex: `audios`).
- `SPEECH_SEGMENTS_DIR`  
  Dossier des segments VAD (ex: `speech_segments`).
- `PROCESSED_AUDIO_DIR`  
  Dossier d’archivage des WAV source traités (ex: `audios_processed`).
- `TRANSCRIPTIONS_DB_PATH`  
  Chemin SQLite des transcriptions.
- `TRANSCRIPTIONS_JSON_OUTPUT`  
  Chemin du fichier d’export JSON.

#### Whisper

- `WHISPER_MODEL_SIZE_OR_PATH`  
  Nom du modèle (ex: `small`, `medium`) ou chemin local vers un modèle.
- `WHISPER_DEVICE`  
  Périphérique d’inférence (`cpu`, `cuda`, etc.).
- `WHISPER_COMPUTE_TYPE`  
  Type de calcul/quantification utilisé par faster-whisper (ex: `int8`, `float16`, `float32`).
- `WHISPER_LANGUAGE`  
  Langue forcée pour la transcription (ex: `fr`, `en`).
- `WHISPER_BEAM_SIZE`  
  Taille du beam search (plus grand = potentiellement plus précis mais plus lent).
- `WHISPER_BEST_OF`  
  Nombre de candidats évalués pour sélectionner le meilleur résultat.
- `WHISPER_PATIENCE`  
  Paramètre de patience du décodage beam search.
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT` (`true/false`, `1/0`, `yes/no`)  
  Si activé, le texte précédent influence le segment courant (cohérence ↑, risque de dérive ↑).
- `WHISPER_TEMPERATURE`  
  Température de décodage (plus élevé = plus de diversité, moins déterministe).
- `WHISPER_COMPRESSION_RATIO_THRESHOLD`  
  Seuil de filtrage des sorties jugées anormales/trop compressées.
- `WHISPER_LOG_PROB_THRESHOLD`  
  Seuil minimal de log-probabilité pour accepter une sortie.
- `WHISPER_NO_SPEECH_THRESHOLD`  
  Seuil de détection « non-parole » au niveau Whisper.

#### VAD

- `VAD_SAMPLE_RATE`  
  Fréquence d’échantillonnage attendue en entrée (Hz), typiquement `16000`.
- `VAD_THRESHOLD`  
  Seuil de détection de parole (plus haut = plus strict).
- `VAD_MIN_SPEECH_DURATION_MS`  
  Durée minimale (ms) pour valider un segment de parole.
- `VAD_MIN_SILENCE_DURATION_MS`  
  Durée minimale (ms) de silence pour couper deux segments.
- `VAD_SPEECH_PAD_MS`  
  Marge (ms) ajoutée avant/après chaque segment détecté.

### 5.3 Exemple d’override

```bash
export PIPELINE_INTERVAL_SECONDS=30
export WHISPER_MODEL_SIZE_OR_PATH=small
export WHISPER_DEVICE=cpu
python pipeline_runner.py --loop
```

---

## 6) Utilisation des scripts unitaires

### VAD seul

```bash
python vad_segment.py
```

### Transcription seule

```bash
python transcribe_segments.py --export-json
```

Options utiles :

```bash
python transcribe_segments.py \
  --input-dir speech_segments \
  --db-path transcriptions/transcriptions.sqlite \
  --json-output transcriptions/transcriptions.json \
  --export-json
```

---

## 7) Exécution en service

### Option A — systemd (recommandé)

Créer `/etc/systemd/system/ecoute-pipeline.service` :

```ini
[Unit]
Description=Ecoute pipeline runner (VAD + transcription)
After=network.target

[Service]
Type=simple
WorkingDirectory=/workspace/ecoute
Environment=PIPELINE_INTERVAL_SECONDS=60
ExecStart=/usr/bin/python3 /workspace/ecoute/pipeline_runner.py --loop
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Puis :

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ecoute-pipeline.service
sudo systemctl status ecoute-pipeline.service
```

### Option B — cron

Exécution périodique en one-shot :

```cron
*/2 * * * * cd /workspace/ecoute && /usr/bin/python3 pipeline_runner.py --once >> /var/log/ecoute_pipeline.log 2>&1
```

Le lockfile protège des chevauchements si un run précédent est encore actif.

---

## 8) Interface Streamlit

`app.py` permet :

- affichage des transcriptions triées,
- filtre texte,
- seuil de confiance,
- lecture des extraits audio (`st.audio`),
- message explicite si le fichier WAV n’existe plus.

---

## 9) Notes opérationnelles

- Le script de capture détecte automatiquement si `ffmpeg` supporte `-rw_timeout`.
- Pour un déploiement stable, privilégier `systemd` + redirection des logs (journalctl).
- Vérifier régulièrement l’espace disque (`audios/`, `speech_segments/`, base SQLite).
