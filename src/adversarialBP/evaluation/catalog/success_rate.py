import pandas as pd

def _success_rate(adversarial_examples):
    """
    Computes success rate for all adversarial_examples.

    Parameters
    ----------
    adversarial_examples:
        All adversarial_examples inclusive nan values.

    Returns
    -------

    """
    total_num_adversarial_examples = len(adversarial_examples)
    successful_adversarial_examples = len(adversarial_examples)
    success_rate = successful_adversarial_examples / total_num_adversarial_examples
    return success_rate


class SuccessRate():
    """
    Computes success rate for the whole recourse method.
    """
    def __init__(self):
        self.columns = ["Success_Rate"]

    def get_evaluation(self, factuals, adversarial_examples):
        rate = _success_rate(adversarial_examples)
        return pd.DataFrame([[rate]], columns=self.columns)
