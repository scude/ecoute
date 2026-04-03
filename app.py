from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

TRANSCRIPTIONS_PATH = Path("transcriptions/transcriptions.json")


@st.cache_data
def load_transcriptions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    df = pd.DataFrame(raw)
    for col in ["timestamp_abs", "confidence", "text", "audio_path"]:
        if col not in df.columns:
            df[col] = None

    df["timestamp_dt"] = pd.to_datetime(df["timestamp_abs"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def confidence_badge(conf: float | None) -> str:
    if conf is None or pd.isna(conf):
        return "⚪ N/A"
    if conf < 0.4:
        return "🔴 Faible"
    if conf < 0.7:
        return "🟠 Moyen"
    return "🟢 Élevé"


def format_timestamp_fr(ts_raw: Any) -> str:
    ts = pd.to_datetime(ts_raw, errors="coerce")
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%d/%m/%Y %H:%M:%S")


def main() -> None:
    st.set_page_config(page_title="Visualiseur de transcriptions", layout="wide")
    st.title("Visualiseur local des transcriptions")

    try:
        df = load_transcriptions(TRANSCRIPTIONS_PATH)
    except FileNotFoundError:
        st.error(
            "Le fichier `transcriptions/transcriptions.json` est introuvable. "
            "Exécutez d'abord l'étape de transcription."
        )
        return
    except json.JSONDecodeError:
        st.error("Le JSON de transcription est invalide. Vérifiez le format du fichier.")
        return

    if df.empty:
        st.warning("Le fichier de transcription est vide.")
        return

    st.sidebar.header("Filtres")
    text_query = st.sidebar.text_input("Filtre texte", value="").strip().lower()

    max_conf = float(df["confidence"].dropna().max()) if df["confidence"].notna().any() else 1.0
    min_conf = float(df["confidence"].dropna().min()) if df["confidence"].notna().any() else 0.0

    conf_threshold = st.sidebar.slider(
        "Seuil de confiance minimum",
        min_value=0.0,
        max_value=max(1.0, max_conf),
        value=max(0.0, min_conf),
        step=0.01,
    )

    filtered = df.copy()
    filtered = filtered[filtered["confidence"].fillna(0.0) >= conf_threshold]

    if text_query:
        filtered = filtered[
            filtered["text"].fillna("").str.lower().str.contains(text_query, regex=False)
        ]

    filtered = filtered.sort_values(by=["timestamp_dt", "timestamp_abs"], na_position="last")

    table_df = filtered[["timestamp_abs", "confidence", "text", "audio_path"]].copy()
    table_df["timestamp_abs"] = table_df["timestamp_abs"].apply(format_timestamp_fr)
    table_df["confidence"] = table_df["confidence"].round(4)

    st.subheader("Tableau des transcriptions")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Écoute par ligne")
    st.caption("Cliquez sur Écouter pour lire le WAV associé.")

    if filtered.empty:
        st.info("Aucune ligne ne correspond aux filtres courants.")
        return

    for idx, row in filtered.reset_index(drop=True).iterrows():
        ts = format_timestamp_fr(row.get("timestamp_abs"))
        text = row.get("text") or ""
        conf = row.get("confidence")
        audio_path_str = str(row.get("audio_path") or "")
        audio_path = Path(audio_path_str)

        left, right = st.columns([6, 1])
        with left:
            st.markdown(
                f"**{idx + 1}.** `{ts}` • {confidence_badge(conf)}  \\\n{text}  \\\n`{audio_path_str}`"
            )
        with right:
            play = st.button("Écouter", key=f"play_{idx}")

        if play:
            if not audio_path_str:
                st.warning(f"Ligne {idx + 1}: chemin audio vide.")
            elif not audio_path.exists():
                st.error(
                    f"Ligne {idx + 1}: fichier audio introuvable: `{audio_path_str}`. "
                    "Vérifiez que les segments WAV existent bien sur disque."
                )
            else:
                st.audio(str(audio_path), format="audio/wav")

        st.divider()


if __name__ == "__main__":
    main()
