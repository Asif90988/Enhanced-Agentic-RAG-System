import logging

logger = logging.getLogger(__name__)

class EnhancedPlanExecuteAgent:
    def __init__(self, routing_agent, planning_agent, react_agent,
                 max_parallel_tasks=3, execution_timeout=60, retry_attempts=2):
        self.routing_agent = routing_agent
        self.planning_agent = planning_agent
        self.react_agent = react_agent
        self.max_parallel_tasks = max_parallel_tasks
        self.execution_timeout = execution_timeout
        self.retry_attempts = retry_attempts

    def execute_task(self, task):
        try:
            logger.info(f"Routing the task: {task}")
            plan = self.routing_agent.route(task)
            logger.info(f"Task routed to plan: {plan}")

            logger.info("Planning execution steps...")
            steps = self.planning_agent.plan(plan)
            logger.info(f"Planned steps: {steps}")

            results = []
            for step in steps:
                logger.info(f"Executing step: {step}")
                result = self.react_agent.react(step)
                results.append(result)
                logger.info(f"Result: {result}")

            summary = self._summarize_results(results)
            logger.info(f"Execution summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            return {"error": str(e)}

    def _summarize_results(self, results):
        return {
            "total_steps": len(results),
            "successful_steps": sum(1 for r in results if r.get("success")),
            "details": results
        }

    def get_agent_statistics(self):
        try:
            routing_stats = self.routing_agent.get_routing_statistics()
            planning_stats = self.planning_agent.get_planning_statistics()
            react_stats = self.react_agent.get_react_statistics()

            return {
                'routing_agent': routing_stats,
                'planning_agent': planning_stats,
                'react_agent': react_stats,
                'execution_config': {
                    'max_parallel_tasks': self.max_parallel_tasks,
                    'execution_timeout': self.execution_timeout,
                    'retry_attempts': self.retry_attempts
                },
                'supported_features': [
                    'multi_step_planning',
                    'parallel_execution',
                    'real_time_data_integration',
                    'dynamic_plan_adjustment',
                    'comprehensive_synthesis'
                ]
            }

        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {}
