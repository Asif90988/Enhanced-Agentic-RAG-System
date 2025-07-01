# agent_response.py

from typing import List, Optional


class AgentResponse:
    def __init__(
        self,
        agent_name: str,
        answer: str,
        confidence: float = 0.0,
        sources: Optional[List[str]] = None,
        reasoning: Optional[str] = None
    ):
        self.agent_name = agent_name
        self.answer = answer
        self.confidence = confidence
        self.sources = sources or []
        self.reasoning = reasoning or ""

    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning": self.reasoning
        }

