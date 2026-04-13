# Écoute — Pipeline de Transcription Audio RTSP

**Écoute** est une solution complète de capture, segmentation et transcription automatisée de flux audio RTSP.

---

## 🏗 Architecture du Système

Le projet est organisé de manière modulaire :
- `ecoute/` : Cœur du système (VAD, Transcription, Stockage, Configuration).
- `ui/` : Interface utilisateur Streamlit.
- `scripts/` : Scripts utilitaires (Capture RTSP, Healthchecks).
- `tests/` : Suite de tests unitaires complète.

---

## 🚀 Démarrage Rapide (Docker Compose)

### 1. Prérequis
- Docker Engine & Docker Compose

### 2. Configuration
```bash
cp .env.sample .env
# Éditez le fichier .env et renseignez RTSP_URL
```

### 3. Lancement
```bash
docker compose up -d --build
```
L'interface est accessible sur : [http://localhost:8501](http://localhost:8501)

---

## 🧪 Tests Unitaires

### Exécuter les tests avec Docker
```bash
docker compose exec pipeline pytest
```

---

## ⚙️ Configuration Avancée

La configuration est gérée dans `ecoute/pipeline_config.py`. Elle peut être surchargée par un fichier `config.yaml` ou des variables d'environnement.

---

## 🛠 Mode Manuel (Développement)

### Installation
```bash
pip install -e .
# Ou installez les dépendances manuellement :
pip install torch torchaudio silero-vad faster-whisper streamlit pandas pyyaml pytest
```

### Commandes
- **Lancer l'orchestrateur** : `python ecoute/pipeline_runner.py --loop`
- **Démarrer l'interface** : `streamlit run ui/app.py`
