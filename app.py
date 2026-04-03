from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from storage import DEFAULT_DB_PATH, SQLiteStorage

TRANSCRIPTIONS_PATH = Path("transcriptions/transcriptions.json")


def resolve_audio_path(audio_path_raw: str) -> Path:
    """
    Resolve audio paths written from different runtimes (host vs container).

    Some transcription rows can contain absolute host paths
    (e.g. /home/.../speech_segments/...). When the UI runs in Docker,
    this path may not exist inside the container. In that case, remap it
    to the local mounted speech segments directory if possible.
    """
    requested = Path(audio_path_raw)
    if requested.exists():
        return requested

    speech_root = os.environ.get("SPEECH_SEGMENTS_DIR", "speech_segments")
    candidate_roots = [
        Path(speech_root),
        Path("/app/speech_segments"),
        Path("speech_segments"),
    ]

    parts = requested.parts
    if "speech_segments" in parts:
        idx = parts.index("speech_segments")
        relative_tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        for root in candidate_roots:
            candidate = root / relative_tail
            if candidate.exists():
                return candidate

    return requested


@st.cache_data
def load_transcriptions_json(path: Path) -> pd.DataFrame:
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


@st.cache_data
def load_transcriptions_sqlite(
    db_path: str,
    text_query: str,
    min_confidence: float,
    page_size: int,
    page_number: int,
) -> tuple[pd.DataFrame, int]:
    storage = SQLiteStorage(Path(db_path))
    offset = page_size * max(page_number - 1, 0)
    rows = storage.query_transcriptions(
        text_query=text_query,
        min_confidence=min_confidence,
        limit=page_size,
        offset=offset,
    )
    total = storage.count_transcriptions(text_query=text_query, min_confidence=min_confidence)

    df = pd.DataFrame(rows)
    for col in ["timestamp_abs", "confidence", "text", "audio_path"]:
        if col not in df.columns:
            df[col] = None

    df["timestamp_dt"] = pd.to_datetime(df["timestamp_abs"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df, total


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

    text_query = st.sidebar.text_input("Filtre texte", value="").strip().lower()
    conf_threshold = st.sidebar.slider(
        "Seuil de confiance minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    page_size = st.sidebar.selectbox("Taille de page", options=[50, 100, 200, 500], index=1)
    page_number = st.sidebar.number_input("Page", min_value=1, value=1, step=1)

    db_exists = DEFAULT_DB_PATH.exists()

    if db_exists:
        try:
            df, total_count = load_transcriptions_sqlite(
                str(DEFAULT_DB_PATH),
                text_query,
                conf_threshold,
                int(page_size),
                int(page_number),
            )
            source_label = f"SQLite ({DEFAULT_DB_PATH})"
        except Exception as exc:
            st.warning(
                f"Lecture SQLite impossible ({exc}). Fallback JSON activé.",
                icon="⚠️",
            )
            db_exists = False

    if not db_exists:
        try:
            df = load_transcriptions_json(TRANSCRIPTIONS_PATH)
            filtered = df[df["confidence"].fillna(0.0) >= conf_threshold]
            if text_query:
                filtered = filtered[
                    filtered["text"].fillna("").str.lower().str.contains(text_query, regex=False)
                ]
            filtered = filtered.sort_values(by=["timestamp_dt", "timestamp_abs"], na_position="last")
            total_count = len(filtered)
            start = (int(page_number) - 1) * int(page_size)
            end = start + int(page_size)
            df = filtered.iloc[start:end]
            source_label = f"JSON ({TRANSCRIPTIONS_PATH})"
        except FileNotFoundError:
            st.error(
                "Aucune base SQLite ni JSON trouvé. Exécutez d'abord l'étape de transcription."
            )
            return
        except json.JSONDecodeError:
            st.error("Le JSON de transcription est invalide. Vérifiez le format du fichier.")
            return

    if df.empty and total_count == 0:
        st.warning("Aucune transcription disponible pour les filtres courants.")
        return

    st.caption(f"Source: {source_label} • Total lignes filtrées: {total_count}")

    table_df = df[["timestamp_abs", "confidence", "text", "audio_path"]].copy()
    table_df["timestamp_abs"] = table_df["timestamp_abs"].apply(format_timestamp_fr)
    table_df["confidence"] = table_df["confidence"].round(4)

    st.subheader("Tableau des transcriptions")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Écoute par ligne")
    st.caption("Cliquez sur Écouter pour lire le WAV associé.")

    if df.empty:
        st.info("Aucune ligne sur cette page.")
        return

    for idx, row in df.reset_index(drop=True).iterrows():
        row_number = (int(page_number) - 1) * int(page_size) + idx + 1
        ts = format_timestamp_fr(row.get("timestamp_abs"))
        text = row.get("text") or ""
        conf = row.get("confidence")
        audio_path_str = str(row.get("audio_path") or "")
        audio_path = resolve_audio_path(audio_path_str) if audio_path_str else Path()

        left, right = st.columns([6, 1])
        with left:
            st.markdown(
                f"**{row_number}.** `{ts}` • {confidence_badge(conf)}  \\\n{text}  \\\n`{audio_path_str}`"
            )
        with right:
            play = st.button("Écouter", key=f"play_{row_number}")

        if play:
            if not audio_path_str:
                st.warning(f"Ligne {row_number}: chemin audio vide.")
            elif not audio_path.exists():
                st.error(
                    f"Ligne {row_number}: fichier audio introuvable: `{audio_path_str}`. "
                    f"Chemin résolu testé: `{audio_path}`. "
                    "Vérifiez que les segments WAV existent bien sur disque."
                )
            else:
                st.audio(str(audio_path), format="audio/wav")

        st.divider()


if __name__ == "__main__":
    main()
