import tdqm

from chm.utils.chm_consts import InferenceMode


class CHMInferenceEngine(object):
    def __init__(self, params_dict):
        self.params_dict = params_dict

        self.overall_batch_num = 0
        self.inference_ds_type = self.params_dict("inference_ds_type", "test")
        self.max_batch_num = None
        self.inference_mode = InferenceMode.BOTH
        self.is_compute_loss = False
        self.force_value_strs = self.set_force_value_strs()

    def set_force_value_strs(self):
        if self.inference_mode == InferenceMode.BOTH:
            self.force_value_strs = ["with_gaze", "without_gaze"]
        elif self.inference_mode == InferenceMode.WITH_GAZE:
            self.force_value_strs = ["with_gaze"]
        elif self.inference_mode == InferenceMode.WITHOUT_GAZE:
            self.force_value_strs = ["without_gaze"]

    def infer(self, module):
        
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
                if self.overall_batch_num > self.max_batch_num:
                    break
            module.inference_step(
                data_batch,
                self.overall_batch_num
                self.force_value_strs,
                self.is_compute_loss
            )
            self.overall_batch_num += 1
