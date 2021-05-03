import torch
import random
import numpy as np
import json
import os

from abc import ABC, abstractmethod
from chm.model.model_wrapper import ModelWrapper
from chm.trainers.cognitive_heatmap_trainer import CHMTrainer
from chm.experiments.chm_inference_engine import CHMInferenceEngine
from collections import OrderedDict


class ChmExperiment(ABC):
    def __init__(self, args, session_hash, training_experiment=False):
        """
        Abstract base class for different CHM experiments for denoising, calibration and awareness estimation

        Parameters:
        -----------
        args: argparse.Namespace
            Contains all args specified in the args_file and any additional arg_setter (specified in the derived classes)

        session_hash: str
            Unique string indicating the sessions id.

        training_experiment: bool
            Bool indicating whether the experiment require training or just inference.
        """

        # create params dict from the args
        self.params_dict = vars(args)

        # select training device
        if self.params_dict["no_cuda"] or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # set random seed
        # if random seed is provided use it, if not create a random seed
        random_seed = self.params_dict["random_seed"] or random.randint(1, 10000)
        print("Random seed used for experiment is ", random_seed)
        self.params_dict["random_seed"] = random_seed  # update the params dict

        # set random seed for numpy and torch
        np.random.seed(self.params_dict["random_seed"])
        torch.manual_seed(self.params_dict["random_seed"])

        # init variables
        self.results_aggregator = {}  # this is a dictionary that saves results on the inference dataset
        self.training_experiment = training_experiment
        # folder to save the results
        self.results_save_folder = os.path.join(os.path.expanduser("~"), "cognitive_heatmap", "results")

        # create model wrapper instance
        self.model_wrapper = ModelWrapper(self.params_dict, session_hash)

    @abstractmethod
    def initialize_functors(self):
        """
        Abstract method to be implemented by the derived class.
        Defines the input_process_functors (used to process the data_input before inference) and output_process_functors (to process and compute metrics on the the inference output)
        """
        pass

    def _perform_experiment(self):
        """
        Depending on the type of experiment, either training or inference is performed.
        """
        if self.training_experiment:
            trainer = CHMTrainer(self.params_dict)
            trainer.fit(self.model_wrapper, ds_type=self.params_dict["inference_ds_type"])
            self.results_aggregator = self.model_wrapper.get_results_aggregator()
        else:
            inference_engine = CHMInferenceEngine(self.params_dict)
            inference_engine.infer(self.model_wrapper)
            self.results_aggregator = self.model_wrapper.get_results_aggregator()

    @abstractmethod
    def perform_experiment(self):
        """
        Abstract method implemented by the derived class. Has to call _perform_experiment() at the end.
        """
        pass

    def shape_results(self, results_aggregator):
        """
        Function to futher shape the results dictionary before saving to disk

        Parameters:
        ---------
        result_aggregator: OrderedDict
            Dict containing the results of a particular experiment. Each key in the dict corresponds to a different metric computed on the output

        Returns:
        -------
        result_aggregator: OrderedDict
            Reshaped results dictionary

        """
        return results_aggregator

    def save_experiment(self, name_values=OrderedDict()):
        """
        Function that saves the results dictionary onto disk.

        Parameters:
        ----------
        name_values: OrderedDict()
            Dictionary containing various strings that are combined to form the filename.

        Returns:
        -------
        None

        """
        assert type(name_values) is OrderedDict
        # create the results folder
        os.makedirs(self.results_save_folder, exist_ok=True)

        # create filename
        results_filename = "experiment"
        for k in name_values:
            results_filename += "_" + k + "_" + str(name_values[k])

        # full path to the results filename
        results_filename = os.path.join(self.results_save_folder, results_filename)
        results_filename += ".json"

        # shape results before saving.
        experiment_results = self.shape_results(self.results_aggregator)

        # save filename
        for k in name_values:
            experiment_results[k] = name_values[k]
        with open(results_filename, "w") as fp:
            json.dump(experiment_results, fp, indent=2)
