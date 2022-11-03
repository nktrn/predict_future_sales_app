import os
from dotenv import load_dotenv

load_dotenv()

neptune_project = os.getenv('project')
neptune_api_token = os.getenv('api_token')
model_id = os.getenv('model_id')
model_name = f"model/{os.getenv('model_name')}"