from Falcon.utils.model_utils import load_model as load_falcon_model
import dotenv
import os


dotenv.load_dotenv("/data2/PromptMIA/.env")
tokenizer, model = load_falcon_model()

workspace = os.getenv("workspace")
result_path = workspace + "Falcon/results/"