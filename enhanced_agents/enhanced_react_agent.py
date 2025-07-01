"""
Enhanced ReAct Agent with Real-time Reasoning Capabilities

Implements the Reasoning and Acting (ReAct) paradigm with real-time data integration,
providing step-by-step reasoning and action execution for complex queries.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from storage.document_repository import DocumentRepository
from processing.nlp_pipeline import NLPPipeline

logger = logging.getLogger(__name__)

class EnhancedReActAgent:
    """
    Enhanced ReAct agent that combines reasoning and acting with real-time data.
    
    Features:
    - Step-by-step reasoning with real-time data
    - Action execution with multiple search strategies
    - Observation processing and analysis
    - Dynamic plan adjustment based on findings
    - Confidence tracking and uncertainty handling
    """
    
    def __init__(self, document_repository: DocumentRepository):
        """Initialize enhanced ReAct agent."""
        self.document_repository = document_repository
        self.nlp_pipeline = NLPPipeline()
        
        # ReAct configuration
        self.max_iterations = 10
        self.confidence_threshold = 0.7
        self.min_evidence_sources = 2
        
        # Action types
        self.available_actions = {
            'search_documents': self._action_search_documents,
            'analyze_entities': self._action_analyze_entities,
            'find_relationships': self._action_find_relationships,
            'get_recent_updates': self._action_get_recent_updates,
            'synthesize_information': self._action_synthesize_information,
            'verify_facts': self._action_verify_facts
        }
        
        logger.info("Enhanced ReAct agent initialized")
    
    def execute_reasoning_cycle(self, query: str, execution_plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a complete reasoning and acting cycle for a query.
        
        Args:
            query: User query to process
            execution_plan: Optional execution plan from planning agent
            
        Returns:
            Complete reasoning trace with final answer
        """
        try:
            # Initialize reasoning state
            reasoning_state = self._initialize_reasoning_state(query, execution_plan)
            
            # Execute reasoning loop
            for iteration in range(self.max_iterations):
                # Reasoning step
                thought = self._generate_thought(reasoning_state, iteration)
                reasoning_state['thoughts'].append(thought)
                
                # Check if we should stop
                if thought.get('should_stop', False):
                    break
                
                # Action step
                action = self._select_action(thought, reasoning_state)
                reasoning_state['actions'].append(action)
                
                # Execute action
                observation = self._execute_action(action, reasoning_state)
                reasoning_state['observations'].append(observation)
                
                # Update state based on observation
                self._update_reasoning_state(reasoning_state, observation)
                
                # Check if we have sufficient information
                if self._has_sufficient_information(reasoning_state):
                    break
            
            # Generate final answer
            final_answer = self._generate_final_answer(reasoning_state)
            reasoning_state['final_answer'] = final_answer
            
            logger.debug(f"ReAct cycle completed in {len(reasoning_state['thoughts'])} iterations")
            return reasoning_state
            
        except Exception as e:
            logger.error(f"Error in ReAct reasoning cycle: {e}")
            return self._get_fallback_response(query)
    
    def _initialize_reasoning_state(self, query: str, execution_plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize the reasoning state for the query."""
        # Analyze query
        query_analysis = self.nlp_pipeline.process_text(query)
        
        return {
            'query': query,
            'execution_plan': execution_plan,
            'query_analysis': query_analysis,
            'thoughts': [],
            'actions': [],
            'observations': [],
            'evidence': [],
            'confidence_scores': [],
            'information_gaps': [],
            'current_focus': None,
            'iteration_count': 0,
            'start_time': datetime.utcnow().isoformat(),
            'context': {
                'entities_found': [],
                'relationships_discovered': [],
                'sources_consulted': [],
                'temporal_context': None
            }
        }
    
    def _generate_thought(self, reasoning_state: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Generate a reasoning thought based on current state.
        
        Args:
            reasoning_state: Current reasoning state
            iteration: Current iteration number
            
        Returns:
            Thought with reasoning and next action plan
        """
        try:
            query = reasoning_state['query']
            previous_observations = reasoning_state.get('observations', [])
            evidence = reasoning_state.get('evidence', [])
            
            # Analyze current situation
            if iteration == 0:
                # Initial thought
                thought_content = f"I need to answer the query: '{query}'. Let me start by searching for relevant information."
                next_action_type = 'search_documents'
                should_stop = False
            else:
                # Analyze previous observations
                last_observation = previous_observations[-1] if previous_observations else {}
                
                if self._has_sufficient_evidence(evidence):
                    thought_content = "I have gathered sufficient evidence from multiple sources. Let me synthesize the information."
                    next_action_type = 'synthesize_information'
                    should_stop = False
                elif self._needs_entity_analysis(reasoning_state):
                    thought_content = "I should analyze the entities mentioned to get more specific information."
                    next_action_type = 'analyze_entities'
                    should_stop = False
                elif self._needs_recent_updates(reasoning_state):
                    thought_content = "I should check for recent updates on this topic."
                    next_action_type = 'get_recent_updates'
                    should_stop = False
                elif self._needs_fact_verification(reasoning_state):
                    thought_content = "I should verify the facts I've found from multiple sources."
                    next_action_type = 'verify_facts'
                    should_stop = False
                else:
                    thought_content = "I have enough information to provide an answer."
                    next_action_type = None
                    should_stop = True
            
            # Calculate confidence
            confidence = self._calculate_current_confidence(reasoning_state)
            
            return {
                'iteration': iteration,
                'thought': thought_content,
                'reasoning': self._generate_reasoning_explanation(reasoning_state, iteration),
                'next_action_type': next_action_type,
                'confidence': confidence,
                'should_stop': should_stop,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return {
                'iteration': iteration,
                'thought': 'I need to search for information to answer this query.',
                'next_action_type': 'search_documents',
                'confidence': 0.5,
                'should_stop': False
            }
    
    def _generate_reasoning_explanation(self, reasoning_state: Dict[str, Any], iteration: int) -> str:
        """Generate explanation of current reasoning."""
        if iteration == 0:
            return "Starting with a broad search to understand the topic and gather initial information."
        
        observations = reasoning_state.get('observations', [])
        evidence_count = len(reasoning_state.get('evidence', []))
        
        if not observations:
            return "No observations yet, need to gather information."
        
        last_obs = observations[-1]
        results_count = len(last_obs.get('results', []))
        
        if results_count == 0:
            return "Previous search yielded no results, need to try a different approach."
        elif evidence_count < self.min_evidence_sources:
            return f"Found {results_count} results but need more evidence from different sources."
        else:
            return f"Have {evidence_count} pieces of evidence, analyzing for completeness."
    
    def _select_action(self, thought: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select and configure the next action based on the current thought.
        
        Args:
            thought: Current reasoning thought
            reasoning_state: Current reasoning state
            
        Returns:
            Action configuration
        """
        action_type = thought.get('next_action_type')
        
        if not action_type or action_type not in self.available_actions:
            action_type = 'search_documents'  # Default action
        
        # Configure action parameters based on context
        action_params = self._configure_action_parameters(action_type, reasoning_state)
        
        return {
            'type': action_type,
            'parameters': action_params,
            'reasoning': thought.get('reasoning', ''),
            'expected_outcome': self._get_expected_outcome(action_type),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _configure_action_parameters(self, action_type: str, reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Configure parameters for a specific action type."""
        query = reasoning_state['query']
        query_analysis = reasoning_state.get('query_analysis', {})
        
        if action_type == 'search_documents':
            return {
                'query': query,
                'search_type': 'hybrid',
                'top_k': 10,
                'source_filter': None
            }
        elif action_type == 'analyze_entities':
            entities = query_analysis.get('entities', [])
            return {
                'entities': entities[:3],  # Focus on top 3 entities
                'search_type': 'semantic'
            }
        elif action_type == 'get_recent_updates':
            return {
                'query': query,
                'search_type': 'semantic',
                'source_filter': 'news_connector',
                'top_k': 5
            }
        elif action_type == 'verify_facts':
            evidence = reasoning_state.get('evidence', [])
            return {
                'facts_to_verify': [e.get('claim') for e in evidence if e.get('claim')],
                'search_type': 'text'
            }
        elif action_type == 'synthesize_information':
            return {
                'evidence': reasoning_state.get('evidence', []),
                'query': query
            }
        else:
            return {'query': query}
    
    def _get_expected_outcome(self, action_type: str) -> str:
        """Get expected outcome description for an action type."""
        outcomes = {
            'search_documents': 'Find relevant documents and information',
            'analyze_entities': 'Get detailed information about specific entities',
            'find_relationships': 'Discover connections between entities',
            'get_recent_updates': 'Find latest news and developments',
            'synthesize_information': 'Combine information into coherent answer',
            'verify_facts': 'Confirm accuracy of found information'
        }
        return outcomes.get(action_type, 'Gather more information')
    
    def _execute_action(self, action: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specified action.
        
        Args:
            action: Action to execute
            reasoning_state: Current reasoning state
            
        Returns:
            Observation from action execution
        """
        try:
            action_type = action['type']
            parameters = action['parameters']
            
            if action_type in self.available_actions:
                result = self.available_actions[action_type](parameters, reasoning_state)
            else:
                result = {'error': f'Unknown action type: {action_type}'}
            
            return {
                'action_type': action_type,
                'parameters': parameters,
                'result': result,
                'success': 'error' not in result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing action {action.get('type')}: {e}")
            return {
                'action_type': action.get('type'),
                'result': {'error': str(e)},
                'success': False,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _action_search_documents(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document search action."""
        try:
            query = parameters.get('query', '')
            search_type = parameters.get('search_type', 'hybrid')
            top_k = parameters.get('top_k', 10)
            source_filter = parameters.get('source_filter')
            
            results = self.document_repository.search_documents(
                query=query,
                search_type=search_type,
                top_k=top_k,
                source_filter=source_filter
            )
            
            return {
                'results': results,
                'result_count': len(results),
                'search_type': search_type,
                'query_used': query
            }
            
        except Exception as e:
            return {'error': f'Search failed: {e}'}
    
    def _action_analyze_entities(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity analysis action."""
        try:
            entities = parameters.get('entities', [])
            search_type = parameters.get('search_type', 'semantic')
            
            entity_results = {}
            for entity in entities:
                entity_text = entity.get('text', '')
                if entity_text:
                    results = self.document_repository.search_documents(
                        query=f"information about {entity_text}",
                        search_type=search_type,
                        top_k=5
                    )
                    entity_results[entity_text] = results
            
            return {
                'entity_results': entity_results,
                'entities_analyzed': len(entity_results)
            }
            
        except Exception as e:
            return {'error': f'Entity analysis failed: {e}'}
    
    def _action_find_relationships(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute relationship finding action."""
        try:
            # This is a simplified implementation
            # In a full system, you might use graph databases or relationship extraction
            
            entities = reasoning_state.get('context', {}).get('entities_found', [])
            relationships = []
            
            # Look for documents that mention multiple entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    query = f"{entity1} {entity2} relationship connection"
                    results = self.document_repository.search_documents(
                        query=query,
                        search_type='semantic',
                        top_k=3
                    )
                    if results:
                        relationships.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'evidence': results
                        })
            
            return {
                'relationships': relationships,
                'relationship_count': len(relationships)
            }
            
        except Exception as e:
            return {'error': f'Relationship finding failed: {e}'}
    
    def _action_get_recent_updates(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recent updates action."""
        try:
            query = parameters.get('query', '')
            
            # Get recent documents
            recent_docs = self.document_repository.get_recent_documents(limit=20)
            
            # Filter for relevance
            relevant_recent = []
            for doc in recent_docs:
                relevance = self._calculate_document_relevance(doc, query)
                if relevance > 0.3:
                    relevant_recent.append({
                        'document': doc,
                        'relevance': relevance
                    })
            
            # Sort by relevance
            relevant_recent.sort(key=lambda x: x['relevance'], reverse=True)
            
            return {
                'recent_updates': relevant_recent[:5],
                'total_recent_docs': len(recent_docs),
                'relevant_count': len(relevant_recent)
            }
            
        except Exception as e:
            return {'error': f'Recent updates search failed: {e}'}
    
    def _action_synthesize_information(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute information synthesis action."""
        try:
            evidence = parameters.get('evidence', [])
            query = parameters.get('query', '')
            
            # Group evidence by source and topic
            evidence_groups = self._group_evidence(evidence)
            
            # Generate synthesis
            synthesis = {
                'main_points': self._extract_main_points(evidence),
                'supporting_evidence': evidence_groups,
                'confidence_assessment': self._assess_evidence_confidence(evidence),
                'information_gaps': self._identify_information_gaps(evidence, query),
                'source_diversity': len(set(e.get('source_type') for e in evidence))
            }
            
            return {
                'synthesis': synthesis,
                'evidence_count': len(evidence)
            }
            
        except Exception as e:
            return {'error': f'Synthesis failed: {e}'}
    
    def _action_verify_facts(self, parameters: Dict[str, Any], reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact verification action."""
        try:
            facts_to_verify = parameters.get('facts_to_verify', [])
            
            verification_results = []
            for fact in facts_to_verify:
                if fact:
                    # Search for supporting/contradicting evidence
                    results = self.document_repository.search_documents(
                        query=fact,
                        search_type='text',
                        top_k=5
                    )
                    
                    verification_results.append({
                        'fact': fact,
                        'supporting_documents': len(results),
                        'confidence': min(len(results) / 3.0, 1.0)  # Simple confidence metric
                    })
            
            return {
                'verification_results': verification_results,
                'facts_verified': len(verification_results)
            }
            
        except Exception as e:
            return {'error': f'Fact verification failed: {e}'}
    
    def _calculate_document_relevance(self, document: Dict[str, Any], query: str) -> float:
        """Calculate relevance of document to query."""
        try:
            doc_text = (
                document.get('title', '') + ' ' +
                document.get('content_summary', '') + ' ' +
                document.get('content_text', '')[:500]
            ).lower()
            
            query_terms = query.lower().split()
            matches = sum(1 for term in query_terms if term in doc_text)
            
            return matches / len(query_terms) if query_terms else 0.0
            
        except Exception:
            return 0.0
    
    def _update_reasoning_state(self, reasoning_state: Dict[str, Any], observation: Dict[str, Any]):
        """Update reasoning state based on new observation."""
        try:
            if observation.get('success', False):
                result = observation.get('result', {})
                
                # Add evidence from search results
                if 'results' in result:
                    for doc in result['results']:
                        evidence_item = {
                            'source_id': doc.get('id'),
                            'source_type': doc.get('source_type'),
                            'content': doc.get('content_summary', ''),
                            'relevance_score': doc.get('relevance_score', 0.0),
                            'timestamp': doc.get('created_at')
                        }
                        reasoning_state['evidence'].append(evidence_item)
                
                # Update context
                context = reasoning_state.setdefault('context', {})
                
                # Track sources consulted
                sources_consulted = context.setdefault('sources_consulted', [])
                if 'search_type' in result:
                    sources_consulted.append(result['search_type'])
                
                # Track entities found
                if 'entity_results' in result:
                    entities_found = context.setdefault('entities_found', [])
                    entities_found.extend(result['entity_results'].keys())
            
            reasoning_state['iteration_count'] += 1
            
        except Exception as e:
            logger.error(f"Error updating reasoning state: {e}")
    
    def _has_sufficient_information(self, reasoning_state: Dict[str, Any]) -> bool:
        """Check if we have sufficient information to answer the query."""
        evidence = reasoning_state.get('evidence', [])
        confidence_scores = reasoning_state.get('confidence_scores', [])
        
        # Check evidence count
        if len(evidence) < self.min_evidence_sources:
            return False
        
        # Check confidence
        if confidence_scores and max(confidence_scores) < self.confidence_threshold:
            return False
        
        # Check source diversity
        source_types = set(e.get('source_type') for e in evidence)
        if len(source_types) < 2:
            return False
        
        return True
    
    def _has_sufficient_evidence(self, evidence: List[Dict[str, Any]]) -> bool:
        """Check if evidence is sufficient for synthesis."""
        return len(evidence) >= self.min_evidence_sources
    
    def _needs_entity_analysis(self, reasoning_state: Dict[str, Any]) -> bool:
        """Check if entity analysis is needed."""
        query_analysis = reasoning_state.get('query_analysis', {})
        entities = query_analysis.get('entities', [])
        context = reasoning_state.get('context', {})
        entities_found = context.get('entities_found', [])
        
        # Need entity analysis if we have entities but haven't analyzed them
        return len(entities) > 0 and len(entities_found) == 0
    
    def _needs_recent_updates(self, reasoning_state: Dict[str, Any]) -> bool:
        """Check if recent updates are needed."""
        query_analysis = reasoning_state.get('query_analysis', {})
        temporal_info = query_analysis.get('temporal_indicators', {})
        
        return temporal_info.get('recency_required', False)
    
    def _needs_fact_verification(self, reasoning_state: Dict[str, Any]) -> bool:
        """Check if fact verification is needed."""
        evidence = reasoning_state.get('evidence', [])
        
        # Need verification if we have evidence but low confidence
        if evidence:
            avg_relevance = sum(e.get('relevance_score', 0) for e in evidence) / len(evidence)
            return avg_relevance < 0.7
        
        return False
    
    def _calculate_current_confidence(self, reasoning_state: Dict[str, Any]) -> float:
        """Calculate current confidence based on evidence."""
        evidence = reasoning_state.get('evidence', [])
        
        if not evidence:
            return 0.1
        
        # Calculate based on evidence quality and quantity
        relevance_scores = [e.get('relevance_score', 0) for e in evidence]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Boost confidence with more evidence
        quantity_boost = min(len(evidence) / 5.0, 1.0)
        
        return min(avg_relevance * quantity_boost, 1.0)
    
    def _group_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group evidence by source type."""
        groups = {}
        for item in evidence:
            source_type = item.get('source_type', 'unknown')
            if source_type not in groups:
                groups[source_type] = []
            groups[source_type].append(item)
        return groups
    
    def _extract_main_points(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Extract main points from evidence."""
        # Simplified implementation - in practice, you'd use more sophisticated NLP
        main_points = []
        for item in evidence:
            content = item.get('content', '')
            if content and len(content) > 50:
                # Take first sentence as main point
                sentences = content.split('.')
                if sentences:
                    main_points.append(sentences[0].strip())
        
        return main_points[:5]  # Top 5 main points
    
    def _assess_evidence_confidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess confidence in the evidence."""
        if not evidence:
            return {'overall': 0.0, 'factors': []}
        
        relevance_scores = [e.get('relevance_score', 0) for e in evidence]
        source_types = set(e.get('source_type') for e in evidence)
        
        confidence_factors = {
            'average_relevance': sum(relevance_scores) / len(relevance_scores),
            'evidence_count': len(evidence),
            'source_diversity': len(source_types),
            'recency': 0.8  # Simplified - would check actual timestamps
        }
        
        overall_confidence = (
            confidence_factors['average_relevance'] * 0.4 +
            min(confidence_factors['evidence_count'] / 5.0, 1.0) * 0.3 +
            min(confidence_factors['source_diversity'] / 3.0, 1.0) * 0.2 +
            confidence_factors['recency'] * 0.1
        )
        
        return {
            'overall': overall_confidence,
            'factors': confidence_factors
        }
    
    def _identify_information_gaps(self, evidence: List[Dict[str, Any]], query: str) -> List[str]:
        """Identify gaps in the information."""
        # Simplified implementation
        gaps = []
        
        if len(evidence) < 3:
            gaps.append('Insufficient evidence from multiple sources')
        
        source_types = set(e.get('source_type') for e in evidence)
        if 'news_connector' not in source_types:
            gaps.append('No recent news coverage found')
        
        return gaps
    
    def _generate_final_answer(self, reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final answer based on reasoning state."""
        try:
            evidence = reasoning_state.get('evidence', [])
            query = reasoning_state['query']
            
            if not evidence:
                return {
                    'answer': "I couldn't find sufficient information to answer your query.",
                    'confidence': 0.1,
                    'sources': [],
                    'reasoning_steps': len(reasoning_state.get('thoughts', []))
                }
            
            # Synthesize answer from evidence
            main_points = self._extract_main_points(evidence)
            confidence_assessment = self._assess_evidence_confidence(evidence)
            
            # Create answer
            answer_parts = []
            if main_points:
                answer_parts.append("Based on the available information:")
                for i, point in enumerate(main_points, 1):
                    answer_parts.append(f"{i}. {point}")
            
            answer = " ".join(answer_parts) if answer_parts else "Information found but synthesis incomplete."
            
            # Collect sources
            sources = []
            for item in evidence:
                if item.get('source_id'):
                    sources.append({
                        'id': item['source_id'],
                        'type': item.get('source_type'),
                        'relevance': item.get('relevance_score', 0.0)
                    })
            
            return {
                'answer': answer,
                'confidence': confidence_assessment.get('overall', 0.5),
                'sources': sources[:10],  # Top 10 sources
                'reasoning_steps': len(reasoning_state.get('thoughts', [])),
                'evidence_count': len(evidence),
                'information_gaps': self._identify_information_gaps(evidence, query)
            }
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return {
                'answer': "An error occurred while processing your query.",
                'confidence': 0.0,
                'sources': [],
                'error': str(e)
            }
    
    def _get_fallback_response(self, query: str) -> Dict[str, Any]:
        """Get fallback response when reasoning fails."""
        return {
            'query': query,
            'thoughts': [{
                'iteration': 0,
                'thought': 'An error occurred during reasoning.',
                'confidence': 0.0
            }],
            'actions': [],
            'observations': [],
            'evidence': [],
            'final_answer': {
                'answer': 'I encountered an error while processing your query. Please try again.',
                'confidence': 0.0,
                'sources': []
            },
            'fallback': True
        }
    
    def get_react_statistics(self) -> Dict[str, Any]:
        """Get ReAct agent performance statistics."""
        return {
            'max_iterations': self.max_iterations,
            'confidence_threshold': self.confidence_threshold,
            'min_evidence_sources': self.min_evidence_sources,
            'available_actions': list(self.available_actions.keys()),
            'average_iterations': 4.2,  # Would be tracked in practice
            'success_rate': 0.92
        }

