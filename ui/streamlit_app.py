"""Streamlit chat interface for the Meeting Intelligence System."""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"


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
        st.header("Transcript Upload")
        uploaded_file = st.file_uploader("Upload a meeting transcript", type=["txt"])

        if uploaded_file is not None and st.button("Ingest Transcript"):
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


if __name__ == "__main__":
    main()
