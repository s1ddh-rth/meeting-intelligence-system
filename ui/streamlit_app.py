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


def _is_audio_file(filename: str) -> bool:
    """Check if a filename has an audio extension."""
    return any(filename.lower().endswith(f".{ext}") for ext in AUDIO_EXTENSIONS)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Meeting Intelligence",
        page_icon="🎙️",
        layout="wide",
    )
    st.title("Meeting Intelligence System")

    # --- Sidebar: Upload & Meeting Selection ---
    with st.sidebar:
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
        st.header("Meetings")

        # Fetch and display ingested meetings
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

        # Show meeting details if one is selected
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
                meta = msg["metadata"]
                cols = st.columns(3)
                cols[0].caption(f"Intent: {meta.get('intent', 'N/A')}")
                cols[1].caption(f"Latency: {meta.get('latency_ms', 'N/A')}ms")
                cols[2].caption(f"Sources: {meta.get('num_sources', 0)}")

    # Chat input
    if question := st.chat_input("Ask about your meetings..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer from API
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

                        # Show metadata
                        metadata = {
                            "intent": data.get("intent", "N/A"),
                            "latency_ms": data.get("latency_ms", "N/A"),
                            "num_sources": len(data.get("sources", [])),
                        }
                        cols = st.columns(3)
                        cols[0].caption(f"Intent: {metadata['intent']}")
                        cols[1].caption(f"Latency: {metadata['latency_ms']}ms")
                        cols[2].caption(f"Sources: {metadata['num_sources']}")

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
