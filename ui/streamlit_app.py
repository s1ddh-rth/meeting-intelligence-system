"""Streamlit chat interface for the Meeting Intelligence System."""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"

AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
AUDIO_MIME_MAP = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
}

# Confidence badge styling
_CONFIDENCE_BADGE = {
    "high": ("High", "#28a745"),
    "medium": ("Medium", "#fd7e14"),
    "low": ("Low", "#dc3545"),
}


def _is_audio_file(filename: str) -> bool:
    """Check if a filename has an audio extension."""
    return any(filename.lower().endswith(f".{ext}") for ext in AUDIO_EXTENSIONS)


def _confidence_html(level: str, score: float) -> str:
    """Build an HTML badge for the confidence level."""
    label, color = _CONFIDENCE_BADGE.get(level, ("Unknown", "#6c757d"))
    pct = f"{score * 100:.0f}%" if score > 0 else "N/A"
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">'
        f'{label} ({pct})</span>'
    )


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Meeting Intelligence",
        page_icon="🎙️",
        layout="wide",
    )
    st.title("Meeting Intelligence System")

    # --- Sidebar ---
    with st.sidebar:
        # --- Help / Guide ---
        with st.expander("How to use this app", expanded=False):
            st.markdown("""
**Meeting Intelligence** analyses meeting transcripts and lets you ask
questions about discussions, decisions, and action items.

---

**Uploading a meeting**

- Click **Browse files** below to upload a file.
- **Text transcripts** (`.txt`) — processed instantly.
- **Audio recordings** (`.mp3`, `.wav`, `.m4a`) — transcribed with
  speaker labels, then processed. Audio can take a few minutes on CPU.

**Asking questions**

Type a question in the chat box, for example:
- *"What are the action items?"*
- *"What did Sarah say about the deadline?"*
- *"Summarise the meeting."*
- *"What decisions were made across all meetings?"*

**Understanding the answer**

Each answer shows:
- **Confidence** — how well the retrieved context matches your question.
  - **High** — strong semantic match or direct database lookup.
  - **Medium** — reasonable match; answer is likely accurate.
  - **Low** — weak match; treat the answer with caution.
- **Intent** — how your question was classified (semantic, speaker,
  structured, cross-meeting).
- **Sources** — the transcript excerpts used to generate the answer,
  each with a relevance score (0-1).

**Filtering by meeting**

Use the **Meetings** dropdown to scope questions to a single meeting
or search across all meetings.
""")

        st.divider()

        # --- Upload ---
        st.header("Upload")
        uploaded_file = st.file_uploader(
            "Upload a transcript or audio file",
            type=["txt"] + AUDIO_EXTENSIONS,
        )

        if uploaded_file is not None:
            is_audio = _is_audio_file(uploaded_file.name)
            file_type_label = "audio" if is_audio else "transcript"

            if st.button(f"Ingest {file_type_label.title()}"):
                if is_audio:
                    _ingest_audio(uploaded_file)
                else:
                    _ingest_transcript(uploaded_file)

        st.divider()

        # --- Meeting selection ---
        st.header("Meetings")

        meetings: list[dict] = []
        try:
            resp = httpx.get(f"{API_BASE}/meetings", timeout=10.0)
            if resp.status_code == 200:
                meetings = resp.json()
        except httpx.HTTPError:
            st.warning("API server not reachable")

        meeting_options = ["All meetings"] + [m["meeting_id"] for m in meetings]
        selected = st.selectbox("Filter by meeting", options=meeting_options)
        selected_meeting_id: str | None = None if selected == "All meetings" else selected

        if selected_meeting_id and meetings:
            meeting = next((m for m in meetings if m["meeting_id"] == selected_meeting_id), None)
            if meeting:
                st.caption(f"Speakers: {', '.join(meeting.get('speakers', []))}")
                st.caption(f"Topics: {', '.join(meeting.get('topics', []))}")
                st.caption(f"Chunks: {meeting.get('num_chunks', 0)}")

    # --- Main Panel: Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                _show_answer_metadata(msg["metadata"])

    # Chat input
    if question := st.chat_input("Ask about your meetings..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = httpx.post(
                        f"{API_BASE}/query",
                        json={
                            "question": question,
                            "meeting_id": selected_meeting_id,
                        },
                        timeout=60.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        st.markdown(answer)

                        metadata = {
                            "confidence": data.get("confidence", "medium"),
                            "confidence_score": data.get("confidence_score", 0.0),
                            "intent": data.get("intent", "N/A"),
                            "latency_ms": data.get("latency_ms", "N/A"),
                            "num_sources": len(data.get("sources", [])),
                        }
                        _show_answer_metadata(metadata)

                        # Show sources in expander
                        sources = data.get("sources", [])
                        if sources:
                            with st.expander(f"View {len(sources)} source(s)"):
                                for src in sources:
                                    st.markdown(
                                        f"**{src['speaker']}** "
                                        f"(Meeting: {src['meeting_id']}, "
                                        f"Time: {src.get('timestamp', 'N/A')}, "
                                        f"Score: {src['relevance_score']:.2f})"
                                    )
                                    st.text(src["text"])
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metadata": metadata,
                        })
                    elif response.status_code == 429:
                        error_data = response.json()
                        rate_msg = error_data.get("detail", "Rate limit exceeded. Please wait and try again.")
                        st.warning(rate_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": rate_msg,
                        })
                    else:
                        error_msg = f"Query failed: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                except httpx.ConnectError:
                    st.error("Cannot connect to API server. Is it running?")
                except httpx.ReadTimeout:
                    st.error("Query timed out. The server may be loading models — try again in a moment.")
                except httpx.HTTPError as e:
                    st.error(f"Request failed: {e}")


def _show_answer_metadata(meta: dict) -> None:
    """Render confidence badge and metadata row below an answer."""
    confidence = meta.get("confidence", "medium")
    confidence_score = meta.get("confidence_score", 0.0)

    st.markdown(
        _confidence_html(confidence, confidence_score),
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].caption(f"Intent: {meta.get('intent', 'N/A')}")
    cols[1].caption(f"Latency: {meta.get('latency_ms', 'N/A')}ms")
    cols[2].caption(f"Sources: {meta.get('num_sources', 0)}")


def _ingest_transcript(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    """Ingest a text transcript file via the /ingest endpoint."""
    with st.spinner("Processing transcript..."):
        try:
            response = httpx.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")},
                timeout=120.0,
            )
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Ingested **{result['meeting_id']}**: "
                    f"{result['chunks_created']} chunks, "
                    f"{result['action_items_count']} action items"
                )
            else:
                st.error(f"Ingestion failed: {response.text}")
        except httpx.ConnectError:
            st.error("Cannot connect to API server. Is it running?")
        except httpx.HTTPError as e:
            st.error(f"Ingestion request failed: {e}")


def _ingest_audio(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    """Ingest an audio file via the /ingest/audio endpoint."""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    mime = AUDIO_MIME_MAP.get(ext, "application/octet-stream")

    progress = st.progress(0, text="Uploading audio...")
    try:
        progress.progress(10, text="Transcribing audio (this may take several minutes on CPU)...")
        response = httpx.post(
            f"{API_BASE}/ingest/audio",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), mime)},
            timeout=600.0,
        )
        progress.progress(90, text="Finalising...")

        if response.status_code == 200:
            result = response.json()
            progress.progress(100, text="Done!")
            st.success(
                f"Transcribed & ingested **{result['meeting_id']}**: "
                f"{result['chunks_created']} chunks, "
                f"{result['action_items_count']} action items\n\n"
                f"Transcript saved as `{result['transcript_filename']}`"
            )
            transcript_text = result.get("transcript_text", "")
            if transcript_text:
                st.download_button(
                    label="Download Transcript",
                    data=transcript_text,
                    file_name=result["transcript_filename"],
                    mime="text/plain",
                )
        else:
            progress.empty()
            detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
            st.error(f"Audio ingestion failed: {detail}")
    except httpx.ConnectError:
        progress.empty()
        st.error("Cannot connect to API server. Is it running?")
    except httpx.ReadTimeout:
        progress.empty()
        st.error("Audio processing timed out. The file may be too large for CPU transcription.")
    except httpx.HTTPError as e:
        progress.empty()
        st.error(f"Audio ingestion request failed: {e}")


if __name__ == "__main__":
    main()
