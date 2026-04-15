from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ecoute.monitoring import get_monitoring_snapshot
from ecoute.storage import DEFAULT_DB_PATH, SQLiteStorage
from ecoute.pipeline_config import load_config

TRANSCRIPTIONS_PATH = Path("transcriptions/transcriptions.json")
CSS_FILE = Path(__file__).parent / "styles" / "navigation.css"

def inject_local_css(css_file: Path) -> None:
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def resolve_audio_path(audio_path_raw: str) -> Path:
    requested = Path(audio_path_raw)
    if requested.exists():
        return requested
    speech_root = os.environ.get("SPEECH_SEGMENTS_DIR", "speech_segments")
    candidate_roots = [Path(speech_root), Path("/app/speech_segments"), Path("speech_segments")]
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
def load_transcriptions_sqlite(
    db_path: str, 
    text_query: str, 
    min_confidence: float | None, 
    page_size: int, 
    page_number: int,
    sort_desc: bool = False
) -> tuple[pd.DataFrame, int]:
    config = load_config()
    banned_phrases = config.get("whisper", {}).get("banned_phrases", [])
    
    storage = SQLiteStorage(Path(db_path), banned_phrases=banned_phrases)
    offset = page_size * max(page_number - 1, 0)
    rows = storage.query_transcriptions(
        text_query=text_query, 
        min_confidence=min_confidence, 
        limit=page_size, 
        offset=offset,
        sort_desc=sort_desc
    )
    total = storage.count_transcriptions(text_query=text_query, min_confidence=min_confidence)
    df = pd.DataFrame(rows)
    
    # Assurer la présence des colonnes nécessaires
    cols = ["timestamp_abs", "confidence", "text", "audio_path", "segment_start_sec", "vad_start_ms", "vad_end_ms"]
    for col in cols:
        if col not in df.columns: df[col] = None
        
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_abs"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["segment_start_sec"] = pd.to_numeric(df["segment_start_sec"], errors="coerce").fillna(0)
    return df, total

def confidence_badge(conf: float | None) -> str:
    if conf is None or pd.isna(conf): return "⚪ N/A"
    
    percent = int(conf * 100)
    if conf < 0.4: return f"🔴 Faible ({percent}%)"
    if conf < 0.7: return f"🟠 Moyen ({percent}%)"
    return f"🟢 Élevé ({percent}%)"

def format_timestamp_fr(ts_raw: Any) -> str:
    ts = pd.to_datetime(ts_raw, errors="coerce")
    return ts.strftime("%d/%m/%Y %H:%M:%S") if not pd.isna(ts) else "N/A"

def render_health_indicator(snapshot: dict[str, Any]) -> None:
    healthy = snapshot.get("overall_healthy", False)
    label = "🟢 Santé OK" if healthy else "🔴 Problème"
    
    _, col_btn = st.columns([8, 2])
    with col_btn:
        if st.button(label, use_container_width=True, help="Voir le monitoring détaillé"):
            st.session_state["nav_menu"] = "Monitoring"
            st.rerun()

def render_monitoring_details(snapshot: dict[str, Any]) -> None:
    st.header("État du système")
    capture = snapshot["capture"]
    pipeline = snapshot["pipeline"]
    pending = snapshot["pending"]
    storage = snapshot["storage"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capture", "🟢 Active" if capture.get("healthy") else "🔴 Stale")
    c2.metric("Pipeline", "🟢 OK" if pipeline.get("healthy") else "🔴 KO")
    c3.metric("Âge dernier run", pipeline["age_human"])
    c4.metric("Backlog files", str(pending["total"]))

    if not capture.get("healthy"):
        st.error(f"⚠️ Capture inactive depuis {capture['age_human']} (max {capture['max_age_seconds']}s)")
    if not pipeline.get("healthy"):
        st.error(f"⚠️ Pipeline en retard ou en échec. Âge: {pipeline['age_human']}")
    
    st.subheader("Stockage")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total", storage["total_human"])
    s2.metric("Utilisé", storage["used_human"])
    s3.metric("Libre", storage["free_human"])
    s4.metric("Taux", f"{storage['used_percent']:.1f}%")

    folder_df = pd.DataFrame([{"Dossier": n, "Taille": d["human"]} for n, d in storage["folder_sizes"].items()])
    st.dataframe(folder_df, use_container_width=True, hide_index=True)

def render_transcriptions_tab() -> None:
    st.header("Transcriptions")
    with st.sidebar:
        st.divider()
        st.subheader("Filtres")
        text_query = st.text_input("Recherche", value="").strip().lower()
        
        conf_threshold = st.slider(
            "Confiance min.", 
            0.0, 1.0, 0.0, 0.01,
            format="%.2f"
        )
        if conf_threshold == 0.0:
            st.caption("Filtre de confiance : **Désactivé**")
        
        sort_order = st.selectbox(
            "Ordre chronologique",
            ["Plus récents en premier", "Plus anciens en premier"],
            index=0
        )
        sort_desc = (sort_order == "Plus récents en premier")

        page_size = st.selectbox("Lignes par page", [50, 100, 200], index=1)
        page_number = st.number_input("Page", min_value=1, value=1)

    db_path = str(DEFAULT_DB_PATH)
    if not DEFAULT_DB_PATH.exists():
        st.error("Base de données introuvable. Lancez le pipeline.")
        return

    df, total = load_transcriptions_sqlite(
        db_path, 
        text_query, 
        conf_threshold if conf_threshold > 0 else None, 
        int(page_size), 
        int(page_number),
        sort_desc=sort_desc
    )
    if df.empty:
        st.warning("Aucun résultat.")
        return

    st.caption(f"Total: {total} lignes")
    
    # Calcul des timestamps exacts par ligne
    df["line_ts_dt"] = df["timestamp_dt"] + pd.to_timedelta(df["segment_start_sec"], unit='s')

    # Regroupement par segment (audio_path)
    groups = []
    current_path = None
    for _, row in df.iterrows():
        if row["audio_path"] != current_path:
            current_path = row["audio_path"]
            groups.append({
                "audio_path": current_path,
                "rows": [],
                "base_ts": row["timestamp_dt"],
                "vad_start_ms": row["vad_start_ms"],
                "vad_end_ms": row["vad_end_ms"]
            })
        groups[-1]["rows"].append(row)

    for group in groups:
        with st.container():
            # Préparation des données du segment
            rows = group["rows"]
            confidences = [r["confidence"] for r in rows if pd.notna(r["confidence"])]
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            
            # Calcul de la plage horaire du segment
            start_dt = group["base_ts"]
            duration_sec = (group["vad_end_ms"] - group["vad_start_ms"]) / 1000.0 if (pd.notna(group["vad_start_ms"]) and pd.notna(group["vad_end_ms"])) else 0
            
            if pd.notna(start_dt):
                end_dt = start_dt + pd.to_timedelta(duration_sec, unit='s')
                date_part = start_dt.strftime("%d/%m/%Y")
                time_range = f"{start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}"
                title_str = f"**{date_part} {time_range}** • {confidence_badge(avg_conf)}"
            else:
                title_str = f"Segment inconnu • {confidence_badge(avg_conf)}"
            
            # Affichage du titre du segment
            st.markdown(title_str)
            
            # Affichage des lignes de texte
            for r in rows:
                l_ts = r["line_ts_dt"].strftime("%H:%M:%S") if pd.notna(r["line_ts_dt"]) else "??:??:??"
                st.markdown(f"**{l_ts}** - {r['text']}")
            
            # Lecteur audio
            audio_path = resolve_audio_path(str(group["audio_path"] or ""))
            if audio_path.exists():
                st.audio(str(audio_path), format="audio/wav")
            else:
                st.caption(f"Audio manquant : `{audio_path.name}`")
            
            st.divider()

def main() -> None:
    st.set_page_config(page_title="Écoute", layout="wide", initial_sidebar_state="expanded")
    inject_local_css(CSS_FILE)
    
    if "nav_menu" not in st.session_state:
        st.session_state["nav_menu"] = "Transcriptions"

    snapshot = get_monitoring_snapshot()
    render_health_indicator(snapshot)

    with st.sidebar:
        st.title("🎧 Écoute")
        selected = st.radio(
            "",
            ["Transcriptions", "Monitoring"],
            key="nav_menu"
        )

    if selected == "Transcriptions":
        render_transcriptions_tab()
    else:
        render_monitoring_details(snapshot)

if __name__ == "__main__":
    main()
