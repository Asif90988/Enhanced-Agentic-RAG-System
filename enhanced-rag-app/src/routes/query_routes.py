from flask import Blueprint, request, jsonify
from enhanced_agents.enhanced_routing_agent import RoutingAgent

query_routes = Blueprint('query_routes_bp', __name__)

# ✅ Create routing agent once
routing_agent = RoutingAgent()

@query_routes.route('/query', methods=['POST'])
def route_query():
    try:
        data = request.get_json()
        question = data.get("query", "")
        
        # ✅ Use the routing agent to get the response
        response = routing_agent.route_query(question)


        return jsonify({
            "answer": response.answer,
            "source_documents": response.sources,
            "agent_used": response.agent_name
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500
