# Écoute : Pipeline de Transcription Audio RTSP

**Écoute** est une solution complète de capture, segmentation et transcription automatisée de flux audio RTSP. Le projet s'appuie sur des technologies de pointe comme **Silero VAD** pour la détection d'activité vocale et **Faster-Whisper** pour une transcription performante, le tout piloté par une interface moderne sous **Streamlit**.

---

## 🏗 Architecture du Système

Le pipeline de données est structuré en quatre étapes clés :

1.  **Capture Audio** : Extraction continue du flux RTSP vers des fichiers WAV (`capture_rtsp_audio.sh`).
2.  **Segmentation VAD** : Détection automatique des zones de parole et découpage en segments optimisés (`vad_segment.py`).
3.  **Transcription IA** : Conversion de la parole en texte via Whisper avec stockage en base SQLite (`transcribe_segments.py`).
4.  **Interface Utilisateur** : Visualisation, filtrage et écoute des segments transcrits via une application Web (`app.py`).

L'orchestrateur **`pipeline_runner.py`** assure la synchronisation et la robustesse des étapes de traitement.

---

## 🚀 Démarrage Rapide (Docker Compose)

La méthode recommandée pour déployer **Écoute** est d'utiliser Docker Compose, qui encapsule toutes les dépendances (FFmpeg, Python, modèles IA).

### 1. Prérequis
- Docker Engine & Docker Compose

### 2. Configuration
Copiez le fichier d'exemple et configurez votre URL RTSP :
```bash
cp .env.sample .env
# Éditez le fichier .env et renseignez RTSP_URL
```

### 3. Lancement
```bash
docker compose up -d --build
```
L'interface est alors accessible sur : [http://localhost:8501](http://localhost:8501)

---

## 🧪 Tests Unitaires

Le projet dispose d'une suite de tests complète pour garantir la fiabilité du traitement audio et de la gestion des données.

### Exécuter les tests avec Docker
```bash
docker compose exec pipeline pytest
```

### Exécuter les tests localement
Si vous avez un environnement Python configuré :
```bash
pip install pytest
pytest
```

Les tests couvrent :
- La persistence et les requêtes **SQLite**.
- La logique de **VAD** et le découpage audio.
- Le parsing des métadonnées de **transcription**.
- La gestion de la **configuration** et des variables d'environnement.
- Les utilitaires de **monitoring** système.

---

## ⚙️ Configuration Avancée

La configuration est gérée de manière hiérarchique :
1.  **Variables d'environnement** (Priorité haute, idéal pour Docker)
2.  **Fichier `config.yaml`** (Configuration statique)
3.  **Valeurs par défaut** (Internes au code)

### Variables clés du `.env`
| Variable | Description | Par défaut |
| :--- | :--- | :--- |
| `RTSP_URL` | URL de la source audio RTSP | *Requis* |
| `WHISPER_MODEL_SIZE_OR_PATH` | Taille du modèle (tiny, small, medium, large-v3) | `medium` |
| `WHISPER_DEVICE` | Périphérique de calcul (`cpu` ou `cuda`) | `cpu` |
| `PIPELINE_INTERVAL_SECONDS` | Délai entre deux cycles de traitement | `60` |

---

## 🛠 Mode Manuel & Développement

Pour exécuter les composants individuellement hors Docker :

### Installation des dépendances
```bash
pip install torch torchaudio silero-vad faster-whisper streamlit pandas pyyaml pytest
```

### Commandes utiles
- **Lancer l'orchestrateur en boucle** : `python pipeline_runner.py --loop`
- **Lancer un traitement unique** : `python pipeline_runner.py --once`
- **Démarrer l'interface UI** : `streamlit run app.py`

---

## 📈 Monitoring et Santé

Le système inclut un mécanisme de surveillance intégré :
- **Heartbeat** : Le service de capture met à jour `runtime/capture_heartbeat.json` en temps réel.
- **Healthcheck** : Docker surveille automatiquement l'état de la capture via `./scripts/health_capture.sh`.
- **Tableau de bord** : L'onglet "Monitoring" de l'UI Streamlit affiche l'état des services, le backlog de fichiers et l'utilisation du stockage disque.

---

## 📂 Structure des Dossiers
- `audios/` : Fichiers WAV bruts en cours de capture.
- `speech_segments/` : Segments extraits par le VAD.
- `transcriptions/` : Base de données SQLite et exports JSON.
- `runtime/` : Fichiers d'état et verrous de processus.
