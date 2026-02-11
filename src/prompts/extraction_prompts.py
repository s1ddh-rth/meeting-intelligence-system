"""Prompts for structured extraction from meeting transcripts."""

EXTRACTION_SYSTEM_PROMPT = """You are a precise meeting analyst. Extract structured information from meeting transcripts accurately. Return ONLY valid JSON with no markdown formatting, no code fences, and no explanation."""

EXTRACTION_USER_PROMPT = """Analyse the following meeting transcript and extract structured information.

TRANSCRIPT:
{transcript}

Return a JSON object with the following structure:
{{
  "action_items": [
    {{"assignee": "name", "task": "description", "deadline": "date or timeframe or null"}}
  ],
  "decisions": [
    {{"decision": "what was decided", "context": "why/how", "decided_by": ["names"]}}
  ],
  "topics": ["topic1", "topic2"],
  "summary": "2-3 sentence summary of the meeting",
  "speakers": ["speaker1", "speaker2"]
}}

IMPORTANT rules for action item deadlines:
- Set deadline to null (not a string "null") when there is NO specific date or timeframe.
- Conditional phrases like "when ready", "once X is done", "when available", "as soon as possible", "once Tom has quotes" are NOT deadlines — set deadline to null for these.
- Only use a deadline value when there is an explicit date (e.g. "Friday", "Jan 30", "end of sprint") or a concrete timeframe (e.g. "by next week", "within 2 days").
- Include the condition in the task description instead (e.g. task: "Review proposal once Tom has quotes").

Return ONLY valid JSON. No markdown, no explanation."""
