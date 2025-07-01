class EnhancedReActAgent:
    def __init__(self, document_repository):
        self.document_repository = document_repository

    def run(self, query):
        """
        Execute a simple ReAct-style response:
        - Use the document repository to search
        - Return a placeholder decision based on search results
        """
        documents = self.document_repository.search(query)
        if documents:
            return f"ReActAgent: Found {len(documents)} document(s) related to '{query}'."
        return "ReActAgent: No relevant documents found."
