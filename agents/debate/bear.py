from agents.debate.base_debate_agent import DebateAgent

class BearAgent(DebateAgent):
    name = "bear_agent"
    description = "Argues the case for holding or reducing exposure"
    stance_prompt = """You are a bearish investment analyst. Given a query and
    retrieved evidence, argue the strongest case for decline/selling/shorting.
    Only make claims you can support with the evidence provided."""