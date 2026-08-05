from debate.base_debate_agent import DebateAgent

class BullAgent(DebateAgent):
    name = "bull_agent"
    description = "Argues the case for buying, holding, or increasing exposure"
    stance_prompt = """You are a bullish investment analyst. Given a query and
    retrieved evidence, argue the strongest case for growth, buying, or holding.
    Only make claims you can support with the evidence provided."""
