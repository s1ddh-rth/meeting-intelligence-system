"""Prompts for query intent classification."""

CLASSIFICATION_SYSTEM_PROMPT = """You are a query classifier for a meeting intelligence system. Classify user questions accurately into exactly one category."""

CLASSIFICATION_USER_PROMPT = """Classify the following user question about meeting transcripts into exactly one category:

- STRUCTURED: Questions about action items, decisions, todos, deadlines, meeting summaries, attendees, or speakers list
  (e.g., "What are the action items?", "What decisions were made?", "Summarize the meeting", "Who attended the meeting?", "Who were the speakers?")
- SPEAKER: Questions about what a specific person said, did, or whether they participated
  (e.g., "What did Sarah say about...", "What were Tom's concerns?", "Was Tom in the meeting?", "Did Sarah speak?")
- SEMANTIC: General questions about topics, discussions, or concepts
  (e.g., "What was discussed about the API?", "Tell me about the budget discussion")
- CROSS_MEETING: Questions comparing or referencing multiple meetings
  (e.g., "How has the timeline changed?", "Compare priorities across meetings")

Question: {question}

Respond with ONLY the category name (STRUCTURED, SPEAKER, SEMANTIC, or CROSS_MEETING)."""
