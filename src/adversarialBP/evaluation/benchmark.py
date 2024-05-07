import timeit
from typing import List
import torch
import numpy as np
import pandas as pd

class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate multiple adversarial examples with
    the given generation method.

    Parameters
    ----------
    mlmodel: 
        Black Box model we want to trick into making an incorrect prediction
    generation_method: 
        Generation method we want to benchmark.
    factuals:
        Instances for which we want to find _adversarial_examples.
    """

    def __init__(
        self,
        dataset_manager,
        mlmodel,
        generation_method,
        factuals, 
        target
    ):
        self.dataset_manager = dataset_manager
        self.mlmodel = mlmodel
        self._generation_method = generation_method
        self._factuals = np.array(factuals.squeeze())
        start = timeit.default_timer()
        print('generating the adversarial examples:')
        self._adversarial_examples = np.array(torch.tensor(generation_method.get_adversarial_examples(factuals, target)))
        stop = timeit.default_timer()
        self.timer = stop - start

    def run_benchmark(self, measures) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Parameters
        ----------
        measures : List[Evaluation]
            List of Evaluation measures that will be computed.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            measure.get_evaluation(
                factuals=self._factuals, adversarial_examples=self._adversarial_examples
            )
            for measure in measures
        ]
        factual_list = np.argmax(self._factuals, axis=1).tolist()
        argmax_cfs = np.argmax(self._adversarial_examples, axis=2)
        adv_list = [row.tolist() for row in argmax_cfs]

        # Create DataFrames from the lists of lists
        df1 = {'factual': [factual_list]*self._adversarial_examples.shape[0], 'counterfactual': adv_list}
        df1 = pd.DataFrame(df1)
        # Concatenate the DataFrames along the columns
        output = pd.concat(pipeline, axis=1)
        result_df = pd.concat([df1, output], axis=1)
        
        return result_df