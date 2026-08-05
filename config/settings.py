from dotenv import load_dotenv
import os

load_dotenv()

MODEL_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# add more settings here as we need them