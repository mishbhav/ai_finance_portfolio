from typing import Any, Dict
from agents.base import Agent

class DataAnalystAgent(Agent):
    """
    Concrete implementation stub for processing financial ingestion datasets.
    """
    def __init__(self):
        # Explicitly satisfies constructor requirements of the base contract
        super().__init__(
            name="DataAnalyst",
            description="Ingests historical pricing feeds to compute baseline indicator values."
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Concrete implementation of the required abstract method.
        """
        # Diagnostic print to verify execution flow during multi-agent tests
        print(f"[{self.name}] Activating workflow for keys: {list(input_data.keys())}")
        
        return {
            "agent": self.name,
            "status": "stub_success",
            "output": {
                "extracted_indicators": ["SMA_20", "RSI_14"],
                "data_quality": "nominal"
            }
        }
