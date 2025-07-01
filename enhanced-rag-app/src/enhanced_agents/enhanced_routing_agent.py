from repository.document_repository import DocumentRepository
from models.agent_response import AgentResponse

class RoutingAgent:
    def __init__(self):
        print("[RoutingAgent] Initialized successfully.")
        self.document_repository = DocumentRepository()

    def route_query(self, question: str) -> AgentResponse:
        documents = self.document_repository.search_documents(question)
        answer = f"This is a routed response to the question: '{question}'"
        sources = [doc.get("source", "unknown") for doc in documents]
        return AgentResponse(answer=answer, sources=sources, agent_name="RoutingAgent")
