"""LLM-based structured extraction from meeting transcripts."""

from __future__ import annotations

import json

import structlog

from src.llm.provider import LLMProvider
from src.models.schemas import (
    ActionItem,
    Decision,
    MeetingExtractions,
    Utterance,
)
from src.prompts.extraction_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
)

logger = structlog.get_logger()


class StructuredExtractor:
    """Extracts action items, decisions, topics, and summaries from transcripts.

    Uses an LLM to analyse the full transcript and return structured data.
    Handles parsing errors gracefully with one retry, then stores partial results.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def extract(
        self, utterances: list[Utterance], meeting_id: str
    ) -> MeetingExtractions:
        """Extract structured data from a list of utterances.

        Args:
            utterances: Parsed transcript utterances.
            meeting_id: Identifier for this meeting.

        Returns:
            MeetingExtractions with action items, decisions, topics, and summary.
        """
        # Build full transcript text for the LLM
        transcript_text = self._format_transcript(utterances)
        speakers = list({u.speaker for u in utterances})

        prompt = EXTRACTION_USER_PROMPT.format(transcript=transcript_text)

        # Try extraction with one retry on parse failure
        for attempt in range(2):
            try:
                response = await self._llm.generate(
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                extractions = self._parse_response(response.content, meeting_id, speakers)
                logger.info(
                    "extraction_complete",
                    meeting_id=meeting_id,
                    action_items=len(extractions.action_items),
                    decisions=len(extractions.decisions),
                    topics=len(extractions.topics),
                )
                return extractions

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "extraction_parse_failed",
                    meeting_id=meeting_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == 1:
                    # Return partial results on second failure
                    logger.error("extraction_failed_returning_partial", meeting_id=meeting_id)
                    return MeetingExtractions(
                        meeting_id=meeting_id,
                        speakers=speakers,
                        summary="Extraction failed — manual review needed.",
                    )

        # Unreachable, but satisfies type checker
        return MeetingExtractions(meeting_id=meeting_id, speakers=speakers)

    def _parse_response(
        self, content: str, meeting_id: str, speakers: list[str]
    ) -> MeetingExtractions:
        """Parse the LLM JSON response into a MeetingExtractions model."""
        # Strip markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)

        return MeetingExtractions(
            meeting_id=meeting_id,
            action_items=[
                ActionItem(
                    assignee=item.get("assignee", "Unknown"),
                    task=item.get("task", ""),
                    deadline=item.get("deadline"),
                )
                for item in data.get("action_items", [])
            ],
            decisions=[
                Decision(
                    decision=dec.get("decision", ""),
                    context=dec.get("context", ""),
                    decided_by=dec.get("decided_by", []),
                )
                for dec in data.get("decisions", [])
            ],
            topics=data.get("topics", []),
            summary=data.get("summary", ""),
            speakers=data.get("speakers", speakers),
        )

    @staticmethod
    def _format_transcript(utterances: list[Utterance]) -> str:
        """Format utterances into a readable transcript string for the LLM."""
        lines: list[str] = []
        for u in utterances:
            if u.timestamp:
                lines.append(f"[{u.timestamp}] {u.speaker}: {u.text}")
            else:
                lines.append(f"{u.speaker}: {u.text}")
        return "\n".join(lines)
