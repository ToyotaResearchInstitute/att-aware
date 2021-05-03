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
        Abstract class for different CHM experiments for denoising, calibration and awareness estimation
        Parameters:
        params_dict: dict

        session_hash: str

        training_experiment: bool
            Bool indicating whether the experiment require training or just inference.
        """

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
        np.random.seed(self.params_dict["random_seed"])
        torch.manual_seed(self.params_dict["random_seed"])

        # init variables
        self.results_aggregator = {}  # this is a dictionary that saves results on the ENTIRE dataset
        self.training_experiment = training_experiment

        # create model wrapper instance
        self.model_wrapper = ModelWrapper(self.params_dict, session_hash)

    @abstractmethod
    def initialize_functors(self):
        pass

    def _perform_experiment(self):
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
        pass

    def shape_results(self, results_aggregator):
        return results_aggregator

    def save_experiment(self, name_values=OrderedDict()):
        assert type(name_values) is OrderedDict
        os.makedirs(self.results_save_folder, exist_ok=True)
        results_filename = "experiment"
        for k in name_values:
            results_filename += "_" + k + "_" + str(name_values[k])
        results_filename = os.path.join(self.results_save_folder, results_filename)
        results_filename += ".json"
        experiment_results = self.shape_results(self.results_aggregator)
        for k in name_values:
            experiment_results[k] = name_values[k]
        with open(results_filename, "w") as fp:
            json.dump(experiment_results, fp, indent=2)
