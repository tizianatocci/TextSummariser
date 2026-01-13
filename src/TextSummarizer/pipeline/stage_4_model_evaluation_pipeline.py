from src.TextSummarizer.config.configuration import ConfigurationManager
from src.TextSummarizer.components.model_evaluation import ModelEvaluation
from src.TextSummarizer.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            logger.exception(f"Exception occurred in initiate_model_evaluation: {e}")
            raise e