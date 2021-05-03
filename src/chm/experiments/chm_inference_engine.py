# Copyright 2020 Toyota Research Institute.  All rights reserved.
import tqdm

from chm.utils.chm_consts import InferenceMode


class CHMInferenceEngine(object):
    """
    Class that defines the inference engine using the CHM.
    """

    def __init__(self, params_dict):
        """
        Parameters:
        ----------
        params_dict: dict
            Dictionary containing the args passed from the experiment script
        """
        self.params_dict = params_dict

        # type of dataset used for inference. Either 'train' or 'test'
        self.inference_ds_type = self.params_dict.get("inference_ds_type", "test")
        # Maximum number of batches to perform inference. Populated by the input_process_dict
        self.max_batch_num = self.params_dict.get("max_inference_num_batches", 20)
        # Inference mode. Determines whether the side-channel input gaze needs to be dropped out or not [WITH_GAZE, WITHOUT_GAZE, BOTH].
        self.inference_mode = InferenceMode.BOTH
        # Boolean which determines whether the loss needs to be computed during inference.
        self.is_compute_loss = False
        self.force_value_strs = self.set_force_value_strs()

    def set_force_value_strs(self):
        """
        Sets the forced_dropout strings according to the inference mode
        """
        if self.inference_mode == InferenceMode.BOTH:
            self.force_value_strs = ["with_gaze", "without_gaze"]
        elif self.inference_mode == InferenceMode.WITH_GAZE:
            self.force_value_strs = ["with_gaze"]
        elif self.inference_mode == InferenceMode.WITHOUT_GAZE:
            self.force_value_strs = ["without_gaze"]

    def infer(self, module):
        """
        Performs inference using CHM model.
        Parameters:
        ----------
        module: ModelWrapper
            This ModelWrapper instance contains the model used for inference.

        Returns:
        --------
        None
        """
        module.inference_engine = self
        # update max_batch_num pull from module.input_process_dict
        if "max_batch_num" in module.input_process_dict and module.input_process_dict["max_batch_num"] is not None:
            self.max_batch_num = module.input_process_dict["max_batch_num"]

        # inference mode - from module.input_process_dict
        if "inference_mode" in module.input_process_dict and module.input_process_dict["inference_mode"] is not None:
            self.inference_mode = module.input_process_dict["inference_mode"]
            # depending on inference mode update the force_value_str
            self.set_force_value_strs()

        # is_compute_loss - from module.input_process_dict
        if "is_compute_loss" in module.input_process_dict:
            if module.input_process_dict["is_compute_loss"]:
                self.is_compute_loss = True

        # set model mode.
        assert "driver_facing" in self.params_dict["dropout_ratio"]
        if self.params_dict["dropout_ratio"]["driver_facing"] < (1 - 5e-2):
            module.train(False)
        else:
            module.train(True)

        # get all dataloaders.
        gaze_dataloaders, awareness_dataloaders, _ = module.get_dataloaders()

        # create the proper dataloader based on and on the inference_ds_type. inference only happens on the gaze and awareness ds's
        if self.inference_ds_type == "train":
            dataloader_tqdm = tqdm.tqdm(
                enumerate(zip(gaze_dataloaders["train"], awareness_dataloaders["train"])),
                desc="inference_train",
            )
        elif self.inference_ds_type == "test":
            dataloader_tqdm = tqdm.tqdm(
                enumerate(zip(gaze_dataloaders["test"], awareness_dataloaders["test"])),
                desc="inference_test",
            )

        # go through the dataloader
        for i, data_batch in dataloader_tqdm:
            if not self.max_batch_num is None:
                if i > self.max_batch_num:
                    break
            module.inference_step(data_batch, i, self.force_value_strs, self.is_compute_loss)

        print("END OF INFERENCE")
