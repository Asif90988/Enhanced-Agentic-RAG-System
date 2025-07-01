from flask import Blueprint, jsonify, request
from enhanced_agents.agent_enhancer import RoutingAgent
from enhanced_agents.custom_agents import CustomConversationalAgent

# Define the blueprint
enhanced_rag_bp = Blueprint('enhanced_rag', __name__)

# Initialize your agent(s)
routing_agent = RoutingAgent()
custom_agent = CustomConversationalAgent()

@enhanced_rag_bp.route('/api/route_query', methods=['POST'])
def route_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        response = routing_agent.route_query(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@enhanced_rag_bp.route('/api/custom_query', methods=['POST'])
def custom_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        response = custom_agent.answer_query(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
