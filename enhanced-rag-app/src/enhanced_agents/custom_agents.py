# enhanced_agents/custom_agents.py

class CustomConversationalAgent:
    def __init__(self, name="Enhanced Agent"):
        self.name = name

    def respond(self, query):
        # Dummy response logic (you can expand later)
        return f"{self.name} received: {query}"

