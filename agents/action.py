from debate.bull import BullAgent
from debate.bear import BearAgent
from debate.risk_parity import RiskParityAgent
from explainer import ExplainerAgent
from judge import JudgeAgent

for cls in [BullAgent, BearAgent, RiskParityAgent, ExplainerAgent, JudgeAgent]:
    agent = cls()
    print(f"{cls.__name__}: name={agent.name!r}, description={agent.description!r}")