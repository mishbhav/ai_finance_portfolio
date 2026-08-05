from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    """
    Abstract Base Class defining the contract for all agents in the ecosystem.
    Ensures structural polymorphism so the orchestrator can call any agent uniformly.
    """
    
    def __init__(self, name: str = None, description: str = None):
        """
        Initializes the agent with required metadata.
        
        Args:
            name: Unique identifier for the agent instance.
            description: Brief summary of the agent's specific responsibility.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the primary logic of the agent.

        Args:
            input_data: A dictionary containing all required context and state variables.

        Returns:
            A dictionary containing the agent's output data and execution status.
        """
        # Minimal placeholder compliance for downstream subclasses
        return {
            "agent": self.name,
            "status": "not_implemented",
            "output": {}
        }
