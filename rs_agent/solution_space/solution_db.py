"""
Solution Database for RS-Agent Solution Space.

Stores professional knowledge on how to use remote sensing tools effectively,
organized as task-solution pairs for Task-Aware Retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SolutionDatabase:
    """
    Database storing expert solutions for remote sensing tasks.

    Each solution entry contains:
    - task_type: Category of remote sensing task
    - task_description: Natural language description of the task
    - solution_steps: Ordered list of tool calls and explanations
    - tools_used: List of tools required
    - tips: Expert tips for this type of task
    """

    def __init__(self, data_dir: str = "data/solutions"):
        self.data_dir = data_dir
        self.solutions: List[Dict[str, Any]] = []
        self._load_solutions()

    def _load_solutions(self):
        """Load solution entries from JSON files."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.warning(f"Solution directory created: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.data_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.solutions.extend(data)
                        elif isinstance(data, dict):
                            self.solutions.append(data)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(self.solutions)} solution entries from {self.data_dir}")

    def add_solution(self, solution: Dict[str, Any]):
        """Add a solution entry to the database."""
        required_fields = ["task_type", "task_description", "solution_steps", "tools_used"]
        for field in required_fields:
            if field not in solution:
                raise ValueError(f"Solution missing required field: {field}")
        self.solutions.append(solution)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all solution entries."""
        return self.solutions

    def get_by_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """Return solutions that use a specific tool."""
        return [s for s in self.solutions if tool_name in s.get("tools_used", [])]

    def get_by_task_type(self, task_type: str) -> List[Dict[str, Any]]:
        """Return solutions for a specific task type."""
        return [s for s in self.solutions if s.get("task_type") == task_type]

    def __len__(self):
        return len(self.solutions)
