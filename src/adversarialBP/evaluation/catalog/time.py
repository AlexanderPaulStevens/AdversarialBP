import pandas as pd

class AvgTime():
    """
    Computes average time for generated adversarial_example
    """

    def __init__(self, hyperparameters= None):
        self.time = hyperparameters["time"]
        self.columns = ["avg_time"]

    def get_evaluation(
        self, factuals: pd.DataFrame, adversarial_examples: pd.DataFrame
    ) -> pd.DataFrame:
        avg_time = self.time / len(adversarial_examples)
        return pd.DataFrame(avg_time, columns=self.columns)
