from debate.base_debate_agent import DebateAgent

class RiskParityAgent(DebateAgent):
    name = "risk_parity_agent"
    description = "Argues the case for holding or increasing exposure"
    stance_prompt = """You are a bullish investment analyst. Given a query and
    retrieved evidence, argue the strongest case for growth/holding/buying.
    Only make claims you can support with the evidence provided."""