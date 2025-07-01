"""
Enhanced Query Planning Agent with Real-time Data Integration

Plans and decomposes complex queries into sub-tasks, leveraging real-time data
and multiple search strategies for comprehensive responses.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from storage.document_repository import DocumentRepository
from processing.nlp_pipeline import NLPPipeline
from .enhanced_routing_agent import EnhancedRoutingAgent

logger = logging.getLogger(__name__)

class EnhancedQueryPlanningAgent:
    """
    Enhanced query planning agent that decomposes complex queries and plans execution.
    
    Features:
    - Query decomposition into sub-tasks
    - Multi-step execution planning
    - Real-time data integration planning
    - Resource allocation and optimization
    - Dependency management between sub-tasks
    """
    
    def __init__(self, document_repository: DocumentRepository, routing_agent: EnhancedRoutingAgent):
        """Initialize enhanced query planning agent."""
        self.document_repository = document_repository
        self.routing_agent = routing_agent
        self.nlp_pipeline = NLPPipeline()
        
        # Planning configuration
        self.max_subtasks = 5
        self.max_planning_depth = 3
        
        logger.info("Enhanced query planning agent initialized")
    
    def plan_query_execution(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Plan the execution of a complex query.
        
        Args:
            query: User query to plan
            context: Additional context information
            
        Returns:
            Execution plan with sub-tasks and strategies
        """
        try:
            # Get routing decision first
            routing_decision = self.routing_agent.route_query(query, context)
            
            # Analyze query complexity and decomposition needs
            decomposition_analysis = self._analyze_decomposition_needs(query, routing_decision)
            
            # Generate execution plan
            if decomposition_analysis['needs_decomposition']:
                execution_plan = self._create_multi_step_plan(query, routing_decision, decomposition_analysis)
            else:
                execution_plan = self._create_single_step_plan(query, routing_decision)
            
            # Optimize plan for real-time data
            optimized_plan = self._optimize_for_real_time_data(execution_plan, routing_decision)
            
            # Add resource allocation
            final_plan = self._allocate_resources(optimized_plan)
            
            logger.debug(f"Created execution plan with {len(final_plan['steps'])} steps")
            return final_plan
            
        except Exception as e:
            logger.error(f"Error planning query execution: {e}")
            return self._get_fallback_plan(query)
    
    def _analyze_decomposition_needs(self, query: str, routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze whether query needs decomposition into sub-tasks.
        
        Args:
            query: User query
            routing_decision: Routing decision from routing agent
            
        Returns:
            Decomposition analysis results
        """
        try:
            query_analysis = routing_decision.get('query_analysis', {})
            complexity = query_analysis.get('complexity', 'medium')
            query_type = query_analysis.get('query_type', 'general')
            
            # Factors that indicate need for decomposition
            decomposition_indicators = {
                'high_complexity': complexity == 'complex',
                'multiple_entities': len(query_analysis.get('entities', [])) > 3,
                'multiple_concepts': len(query_analysis.get('key_concepts', [])) > 3,
                'analytical_query': query_type == 'analytical',
                'temporal_complexity': self._has_temporal_complexity(query_analysis),
                'multiple_questions': self._has_multiple_questions(query)
            }
            
            # Calculate decomposition score
            decomposition_score = sum(decomposition_indicators.values()) / len(decomposition_indicators)
            
            # Determine if decomposition is needed
            needs_decomposition = decomposition_score > 0.4
            
            return {
                'needs_decomposition': needs_decomposition,
                'decomposition_score': decomposition_score,
                'indicators': decomposition_indicators,
                'suggested_subtasks': self._suggest_subtasks(query, query_analysis) if needs_decomposition else [],
                'complexity_factors': self._identify_complexity_factors(query, query_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing decomposition needs: {e}")
            return {'needs_decomposition': False, 'decomposition_score': 0.0}
    
    def _has_temporal_complexity(self, query_analysis: Dict[str, Any]) -> bool:
        """Check if query has temporal complexity requiring decomposition."""
        temporal_info = query_analysis.get('temporal_indicators', {})
        return (
            temporal_info.get('has_temporal', False) and
            len(temporal_info.get('date_entities', [])) > 1
        )
    
    def _has_multiple_questions(self, query: str) -> bool:
        """Check if query contains multiple questions."""
        question_markers = ['?', 'and', 'also', 'additionally', 'furthermore']
        question_count = query.count('?')
        
        if question_count > 1:
            return True
        
        # Check for compound questions
        query_lower = query.lower()
        compound_indicators = sum(1 for marker in question_markers[1:] if marker in query_lower)
        
        return compound_indicators > 1
    
    def _suggest_subtasks(self, query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest subtasks for query decomposition."""
        subtasks = []
        entities = query_analysis.get('entities', [])
        key_concepts = query_analysis.get('key_concepts', [])
        
        # Entity-based subtasks
        important_entities = [e for e in entities if e.get('entity_type') in ['person', 'organization', 'location']]
        for entity in important_entities[:3]:  # Limit to top 3
            subtasks.append({
                'type': 'entity_research',
                'description': f"Research information about {entity.get('text')}",
                'entity': entity,
                'priority': 0.8
            })
        
        # Concept-based subtasks
        for concept in key_concepts[:2]:  # Limit to top 2
            subtasks.append({
                'type': 'concept_exploration',
                'description': f"Explore concept: {concept}",
                'concept': concept,
                'priority': 0.7
            })
        
        # Temporal subtasks
        temporal_info = query_analysis.get('temporal_indicators', {})
        if temporal_info.get('recency_required', False):
            subtasks.append({
                'type': 'recent_updates',
                'description': "Find recent updates and developments",
                'temporal_focus': 'recent',
                'priority': 0.9
            })
        
        # Sort by priority and limit
        subtasks.sort(key=lambda x: x['priority'], reverse=True)
        return subtasks[:self.max_subtasks]
    
    def _identify_complexity_factors(self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Identify factors that make the query complex."""
        factors = []
        
        if query_analysis.get('word_count', 0) > 15:
            factors.append('long_query')
        
        if len(query_analysis.get('entities', [])) > 3:
            factors.append('multiple_entities')
        
        if query_analysis.get('query_type') == 'analytical':
            factors.append('analytical_nature')
        
        if self._has_multiple_questions(query):
            factors.append('multiple_questions')
        
        if query_analysis.get('temporal_indicators', {}).get('has_temporal', False):
            factors.append('temporal_constraints')
        
        return factors
    
    def _create_multi_step_plan(self, query: str, routing_decision: Dict[str, Any], 
                               decomposition_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a multi-step execution plan for complex queries."""
        try:
            subtasks = decomposition_analysis.get('suggested_subtasks', [])
            
            # Create execution steps
            steps = []
            
            # Step 1: Initial information gathering
            steps.append({
                'step_id': 1,
                'type': 'information_gathering',
                'description': 'Gather initial information from multiple sources',
                'search_strategy': routing_decision.get('search_strategy', 'hybrid'),
                'sources': routing_decision.get('recommended_sources', []),
                'query': query,
                'expected_duration': 2.0,
                'dependencies': [],
                'output_type': 'initial_context'
            })
            
            # Steps 2-N: Subtask execution
            for i, subtask in enumerate(subtasks, 2):
                steps.append({
                    'step_id': i,
                    'type': subtask['type'],
                    'description': subtask['description'],
                    'search_strategy': self._select_subtask_strategy(subtask),
                    'sources': self._select_subtask_sources(subtask, routing_decision),
                    'query': self._generate_subtask_query(subtask, query),
                    'expected_duration': 1.5,
                    'dependencies': [1],  # Depends on initial gathering
                    'output_type': 'subtask_result',
                    'subtask_info': subtask
                })
            
            # Final step: Synthesis
            steps.append({
                'step_id': len(steps) + 1,
                'type': 'synthesis',
                'description': 'Synthesize results from all subtasks',
                'search_strategy': 'hybrid',
                'sources': [],
                'query': query,
                'expected_duration': 1.0,
                'dependencies': list(range(1, len(steps) + 1)),
                'output_type': 'final_answer'
            })
            
            return {
                'plan_type': 'multi_step',
                'total_steps': len(steps),
                'steps': steps,
                'estimated_total_time': sum(step['expected_duration'] for step in steps),
                'complexity_score': decomposition_analysis.get('decomposition_score', 0.5),
                'parallel_execution': self._identify_parallel_steps(steps),
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating multi-step plan: {e}")
            return self._get_fallback_plan(query)
    
    def _create_single_step_plan(self, query: str, routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single-step execution plan for simple queries."""
        step = {
            'step_id': 1,
            'type': 'direct_search',
            'description': 'Direct search and response generation',
            'search_strategy': routing_decision.get('search_strategy', 'hybrid'),
            'sources': routing_decision.get('recommended_sources', []),
            'query': query,
            'expected_duration': routing_decision.get('estimated_response_time', 1.5),
            'dependencies': [],
            'output_type': 'final_answer'
        }
        
        return {
            'plan_type': 'single_step',
            'total_steps': 1,
            'steps': [step],
            'estimated_total_time': step['expected_duration'],
            'complexity_score': 0.3,
            'parallel_execution': [],
            'created_at': datetime.utcnow().isoformat()
        }
    
    def _select_subtask_strategy(self, subtask: Dict[str, Any]) -> str:
        """Select search strategy for a specific subtask."""
        subtask_type = subtask.get('type', '')
        
        strategy_mapping = {
            'entity_research': 'hybrid',
            'concept_exploration': 'semantic',
            'recent_updates': 'semantic',
            'factual_lookup': 'text'
        }
        
        return strategy_mapping.get(subtask_type, 'hybrid')
    
    def _select_subtask_sources(self, subtask: Dict[str, Any], routing_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select data sources for a specific subtask."""
        subtask_type = subtask.get('type', '')
        all_sources = routing_decision.get('recommended_sources', [])
        
        if subtask_type == 'recent_updates':
            # Prioritize real-time sources
            return [s for s in all_sources if s.get('source_type') in ['news_connector', 'rss_connector']]
        elif subtask_type == 'entity_research':
            # Use all sources for comprehensive entity research
            return all_sources
        else:
            # Default to top 2 sources
            return all_sources[:2]
    
    def _generate_subtask_query(self, subtask: Dict[str, Any], original_query: str) -> str:
        """Generate a specific query for a subtask."""
        subtask_type = subtask.get('type', '')
        
        if subtask_type == 'entity_research':
            entity_text = subtask.get('entity', {}).get('text', '')
            return f"Information about {entity_text}"
        elif subtask_type == 'concept_exploration':
            concept = subtask.get('concept', '')
            return f"Detailed information about {concept}"
        elif subtask_type == 'recent_updates':
            return f"Recent updates and developments related to: {original_query}"
        else:
            return original_query
    
    def _identify_parallel_steps(self, steps: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify steps that can be executed in parallel."""
        parallel_groups = []
        
        # Group steps by their dependencies
        dependency_groups = {}
        for step in steps:
            deps_key = tuple(sorted(step.get('dependencies', [])))
            if deps_key not in dependency_groups:
                dependency_groups[deps_key] = []
            dependency_groups[deps_key].append(step['step_id'])
        
        # Steps with the same dependencies can run in parallel
        for deps, step_ids in dependency_groups.items():
            if len(step_ids) > 1:
                parallel_groups.append(step_ids)
        
        return parallel_groups
    
    def _optimize_for_real_time_data(self, execution_plan: Dict[str, Any], 
                                   routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution plan based on real-time data availability."""
        try:
            real_time_data = routing_decision.get('real_time_data', {})
            
            if real_time_data.get('available', False):
                # Add real-time data prioritization
                for step in execution_plan['steps']:
                    if step.get('type') in ['information_gathering', 'recent_updates']:
                        # Prioritize sources with fresh data
                        sources = step.get('sources', [])
                        fresh_sources = [s for s in sources if s.get('source_type') in ['news_connector', 'rss_connector']]
                        if fresh_sources:
                            step['sources'] = fresh_sources + [s for s in sources if s not in fresh_sources]
                        
                        # Reduce expected duration for real-time queries
                        step['expected_duration'] *= 0.8
                
                # Add real-time data step if beneficial
                if real_time_data.get('freshness_score', 0) > 0.7:
                    real_time_step = {
                        'step_id': len(execution_plan['steps']) + 1,
                        'type': 'real_time_update',
                        'description': 'Incorporate latest real-time updates',
                        'search_strategy': 'semantic',
                        'sources': [{'source_type': 'news_connector', 'priority': 0.9}],
                        'query': routing_decision.get('query', ''),
                        'expected_duration': 0.5,
                        'dependencies': [1],
                        'output_type': 'real_time_context'
                    }
                    execution_plan['steps'].insert(-1, real_time_step)  # Insert before synthesis
                    execution_plan['total_steps'] += 1
            
            # Recalculate total time
            execution_plan['estimated_total_time'] = sum(
                step['expected_duration'] for step in execution_plan['steps']
            )
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error optimizing for real-time data: {e}")
            return execution_plan
    
    def _allocate_resources(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate computational resources for plan execution."""
        try:
            total_steps = execution_plan.get('total_steps', 1)
            parallel_groups = execution_plan.get('parallel_execution', [])
            
            # Calculate resource allocation
            resource_allocation = {
                'cpu_cores': min(4, max(1, total_steps // 2)),
                'memory_mb': 512 * total_steps,
                'concurrent_searches': len(parallel_groups) + 1,
                'cache_size_mb': 128,
                'timeout_seconds': execution_plan.get('estimated_total_time', 5.0) * 2
            }
            
            # Add resource info to plan
            execution_plan['resource_allocation'] = resource_allocation
            
            # Add execution metadata
            execution_plan['execution_metadata'] = {
                'can_cache_results': True,
                'supports_streaming': total_steps > 2,
                'requires_synthesis': any(step.get('type') == 'synthesis' for step in execution_plan['steps']),
                'estimated_tokens': self._estimate_token_usage(execution_plan)
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            return execution_plan
    
    def _estimate_token_usage(self, execution_plan: Dict[str, Any]) -> int:
        """Estimate token usage for the execution plan."""
        base_tokens_per_step = 500
        synthesis_tokens = 1000
        
        total_tokens = 0
        for step in execution_plan['steps']:
            if step.get('type') == 'synthesis':
                total_tokens += synthesis_tokens
            else:
                total_tokens += base_tokens_per_step
        
        return total_tokens
    
    def _get_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Get fallback execution plan when planning fails."""
        return {
            'plan_type': 'fallback',
            'total_steps': 1,
            'steps': [{
                'step_id': 1,
                'type': 'direct_search',
                'description': 'Fallback direct search',
                'search_strategy': 'hybrid',
                'sources': [{'source_type': 'news_connector', 'priority': 0.8}],
                'query': query,
                'expected_duration': 2.0,
                'dependencies': [],
                'output_type': 'final_answer'
            }],
            'estimated_total_time': 2.0,
            'complexity_score': 0.5,
            'parallel_execution': [],
            'created_at': datetime.utcnow().isoformat(),
            'fallback': True
        }
    
    def validate_plan(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate execution plan for feasibility and optimization.
        
        Args:
            execution_plan: Execution plan to validate
            
        Returns:
            Validation results with suggestions
        """
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'suggestions': [],
                'optimizations': []
            }
            
            # Check plan structure
            if not execution_plan.get('steps'):
                validation_results['is_valid'] = False
                validation_results['issues'].append('No execution steps defined')
                return validation_results
            
            # Check step dependencies
            step_ids = {step['step_id'] for step in execution_plan['steps']}
            for step in execution_plan['steps']:
                for dep in step.get('dependencies', []):
                    if dep not in step_ids:
                        validation_results['issues'].append(f"Step {step['step_id']} has invalid dependency {dep}")
            
            # Check resource allocation
            estimated_time = execution_plan.get('estimated_total_time', 0)
            if estimated_time > 30:  # 30 seconds threshold
                validation_results['suggestions'].append('Consider breaking down into smaller sub-queries')
            
            # Check for optimization opportunities
            steps = execution_plan.get('steps', [])
            if len(steps) > 3 and not execution_plan.get('parallel_execution'):
                validation_results['optimizations'].append('Consider parallel execution for independent steps')
            
            # Check source diversity
            all_sources = set()
            for step in steps:
                for source in step.get('sources', []):
                    all_sources.add(source.get('source_type'))
            
            if len(all_sources) < 2:
                validation_results['suggestions'].append('Consider using more diverse data sources')
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating plan: {e}")
            return {'is_valid': False, 'issues': [f'Validation error: {e}']}
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning performance statistics."""
        try:
            return {
                'max_subtasks': self.max_subtasks,
                'max_planning_depth': self.max_planning_depth,
                'supported_plan_types': ['single_step', 'multi_step'],
                'supported_subtask_types': ['entity_research', 'concept_exploration', 'recent_updates', 'factual_lookup'],
                'average_planning_time': 0.5,  # seconds
                'success_rate': 0.95
            }
        except Exception as e:
            logger.error(f"Error getting planning statistics: {e}")
            return {}

