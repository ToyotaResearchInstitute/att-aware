import os
import torch
import collections
import pickle
import tensorboardX

from chm.utils.trainer_utils import (
    create_model_and_loss_fn,
    load_datasets,
    create_dataloaders,
    save_model,
    parse_data_item,
    parse_data_batch,
    sample_to_device,
    process_and_extract_data_batch,
)


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a CHM model.
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    params_dict : dict
        params dict containing the args from args_file.py
    session_hash : str
        unique hashid for creating logging folder
    """

    def __init__(self, params_dict, session_hash=None):
        super().__init__()
        self.params_dict = params_dict
        self.log_dir = os.path.join(params_dict["log_dir"], session_hash)
        self.training_hash = self.params_dict.get("training_hash", None)
        # create tensorboard logger
        if not self.log_dir is None:
            if self.training_hash is None:
                self.training_hash = self.log_dir
            logger = tensorboardX.SummaryWriter(log_dir=self.log_dir + self.training_hash, comment=self.training_hash)

        self.logger = logger
        # create directory for saving the model
        self.save_model_dir = os.path.join(os.path.expanduser("~"), "cognitive_heatmap", "models", self.training_hash)
        os.makedirs(self.save_model_dir, exist_ok=True)

        if self.params_dict["no_cuda"] or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self.enable_amp = self.params_dict.get("enable_amp", True)
        self.gaze_corruption = self.params_dict.get("gaze_corruption", None)
        self.gaze_correction = self.params_dict.get("gaze_correction", None)
        self.input_process_dict = self.params_dict.get("input_process_dict", None)
        self.output_process_dict = self.params_dict.get("output_process_dict", None)
        self.results_aggregator = {}

        # create model and loss function and put it on the correct device.
        self.model, self.loss_fn = create_model_and_loss_fn(self.params_dict)
        self.loss_fn.to(self.device)
        self.model.to(self.device)

        self.gaze_datasets = self.awareness_datasets = self.pairwise_gaze_datasets = None
        self.gaze_dataset_indices = self.awareness_dataset_indices = self.pairwise_gaze_dataset_indices = None
        # create gaze, awareness and pairwise-gaze datasets
        (self.gaze_datasets, self.awareness_datasets, self.pairwise_gaze_datasets), (
            self.gaze_dataset_indices,
            self.awareness_dataset_indices,
            self.pairwise_gaze_dataset_indices,
        ) = load_datasets(self.params_dict)

        # if use_std_train_test_split save the indices
        self.save_train_test_indices()

        # create gaze, awareness and pairwise-gaze dataloaders
        self.gaze_dataloaders, self.awareness_dataloaders, self.pairwise_gaze_dataloaders = create_dataloaders(
            self.gaze_datasets, self.awareness_datasets, self.pairwise_gaze_datasets, self.params_dict
        )

    def configure_optimizers(self):
        """
        Configure the optimizer, learning rate scheduler and gradient scaler.
        """

        #  extract all model parameters.
        self.optimization_params = self.model.parameters()

        # extract individual modules of the cognitive heat net
        road_facing_network = self.model.fusion_net.map_modules["road_facing"]

        # Check whether encoder parameters need to be frozen
        if self.params_dict["nograd_encoder"]:
            for p in road_facing_network.encoder.parameters():
                p.requires_grad = False
        else:
            for p in road_facing_network.encoder.parameters():
                p.requires_grad = False

            if not self.params_dict["use_s3d"]:
                # for full conv2d only layer3 and layer4 (last two layers) are unfrozen
                for p in road_facing_network.encoder.layered_outputs.layer4.parameters():
                    p.requires_grad = True
                for p in road_facing_network.encoder.layered_outputs.layer3.parameters():
                    p.requires_grad = True
            else:
                # for the s3d encoder, only the last three encoder layers are unfrozen. These are spatio-temporal convolutions using s3D
                for p in road_facing_network.encoder.s3d_net.s3d_net_3.parameters():
                    p.requires_grad = True
                for p in road_facing_network.encoder.s3d_net.s3d_net_2.parameters():
                    p.requires_grad = True
                for p in road_facing_network.encoder.s3d_net.s3d_net_1.parameters():
                    p.requires_grad = True

        # log trainable params from full model info on tensorboard
        self.log_params_info()

        # if param_grad_setter function handle is provided, override the default set of parameters to be optimized.
        # Used in experiments that rely on training
        param_grad_setter = self.params_dict.get("param_grad_setter", None)
        if param_grad_setter is not None:
            self.model, self.optimization_params = param_grad_setter(self.model)

        # Check flag for automatic mixed precision

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.optimization_params),
            lr=self.params_dict.get("learning_rate", 0.0005),
        )
        self.optimizer.zero_grad()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
        self.optim_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.params_dict.get("learning_rate_decay", 0.95)
        )
        return self.optimizer, self.optim_lr_scheduler, self.grad_scaler

    def training_step(self, data_batch, *args):
        # parse all the batches properly.
        overall_batch_num = args[0]
        individual_batch_inputs, awareness_batch_annotation_data = process_and_extract_data_batch(
            data_batch, self.gaze_corruption, self.gaze_correction, self.input_process_dict, self.device
        )
        (
            gaze_batch_input,
            awareness_batch_input,
            pairwise_gaze_batch_input_t,
            pairwise_gaze_batch_input_tp1,
            gaze_aux_info_list,
            awareness_aux_info_list,
            pairwise_gaze_aux_info_list_t,
            pairwise_gaze_aux_info_list_tp1,
            gaze_batch_target,
            awareness_batch_target,
            pairwise_gaze_batch_target_t,
            pairwise_gaze_batch_target_tp1,
            gaze_batch_should_use_batch,
            awareness_batch_should_use_batch,
            pairwise_gaze_batch_should_use_batch_t,
            pairwise_gaze_batch_should_use_batch_tp1,
        ) = individual_batch_inputs

        # ensure force dropout dict is empty during training
        self.model.fusion_net.force_input_dropout = {}

        # Model forwards for all the parsed data items
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            predicted_gaze_output, _, _, _ = self.model.forward(gaze_batch_input)
            predicted_awareness_output, _, _, _ = self.model.forward(awareness_batch_input)

            predicted_pairwise_gaze_t, _, _, should_drop_dicts = self.model.forward(pairwise_gaze_batch_input_t)
            should_drop_indices_dict, should_drop_entire_channel_dict = should_drop_dicts
            predicted_pairwise_gaze_tp1, _, _, should_drop_dicts_tp1 = self.model.forward(
                pairwise_gaze_batch_input_tp1,
                should_drop_indices_dict=should_drop_indices_dict,
                should_drop_entire_channel_dict=should_drop_entire_channel_dict,
            )
            should_drop_indices_dict_tp1, should_drop_entire_channel_dict_tp1 = should_drop_dicts_tp1

            # these asserts are to make sure that both tp and tp1 used the same dropout indices.
            assert (
                should_drop_indices_dict.keys()
                == should_drop_entire_channel_dict.keys()
                == should_drop_indices_dict_tp1.keys()
                == should_drop_entire_channel_dict_tp1.keys()
            )
            for key in should_drop_indices_dict.keys():
                assert torch.all(torch.eq(should_drop_indices_dict[key], should_drop_indices_dict_tp1[key]))
                assert torch.all(
                    torch.eq(should_drop_entire_channel_dict[key], should_drop_entire_channel_dict_tp1[key])
                )

            loss, stats = self.loss_fn.loss(
                predicted_gaze_output,
                gaze_batch_input,
                gaze_batch_target,
                predicted_awareness_output,
                awareness_batch_input,
                awareness_batch_target,
                awareness_batch_annotation_data,
                predicted_pairwise_gaze_t,
                pairwise_gaze_batch_input_t,
                pairwise_gaze_batch_target_t,
                predicted_pairwise_gaze_tp1,
                pairwise_gaze_batch_input_tp1,
                pairwise_gaze_batch_target_tp1,
            )
            import IPython

            IPython.embed(banner1="check after loss")

        if self.output_process_dict is not None:
            output_process_functor = self.output_process_dict["functor"]
            training_output_dict = {}
            training_output_dict["gaze_batch_input"] = gaze_batch_input
            training_output_dict["gaze_aux_info_list"] = gaze_aux_info_list
            training_output_dict["gaze_batch_target"] = gaze_batch_target
            training_output_dict["gaze_data_dict"] = gaze_data_dict
            training_output_dict["gaze_predicted_output"] = predicted_gaze_output

            training_output_dict["awareness_batch_input"] = awareness_batch_input
            training_output_dict["awareness_aux_info_list"] = awareness_aux_info_list
            training_output_dict["awareness_batch_gaze_target"] = awareness_batch_target
            training_output_dict["awareness_batch_annotation_target_data"] = awareness_batch_label_data
            training_output_dict["awareness_predicted_output"] = predicted_awareness_output

            self.results_aggregator = output_process_functor(
                training_output_dict,
                self.output_process_dict["params"],
                overall_batch_num,
                model=self.model,
                experiment_results_aggregator=self.results_aggregator,
            )

        output = {"loss": loss, "stats": stats}
        return output

    def testing_step(self):
        pass

    def visualization_step(self):
        pass

    # getters:
    def get_dataloaders(self):
        return self.gaze_dataloaders, self.awareness_dataloaders, self.pairwise_gaze_dataloaders

    #### logging and saving functions
    def log_params_info(self):
        """
        Log the trainable parameter info to the tensorboardX instance
        """
        params_elements = []
        params_require_grad_elements = []
        params_require_grad_names = []
        for p in self.model.parameters():
            params_elements.append(p.numel())
        for p in self.model.parameters():
            if p.requires_grad:
                params_require_grad_elements.append(p.numel())
        for p in self.model.parameters():
            if p.requires_grad:
                params_require_grad_names.append(p.name)

        self.logger.add_text(tag="model_param_counts", text_string=str(params_elements))
        self.logger.add_text(tag="require_grad_param_counts", text_string=str(params_require_grad_elements))
        self.logger.add_text(tag="require_grad_param_names", text_string=str(params_require_grad_names))

    def save_model(self, overall_batch_num):
        try:
            save_model(
                self.model.module.state_dict(),
                os.path.join(self.save_model_dir, str(overall_batch_num) + "_model.pt"),
            )
        except AttributeError:
            save_model(self.model.state_dict(), os.path.join(self.save_model_dir, str(overall_batch_num) + "_model.pt"))

    def save_train_test_indices(self):
        if self.params_dict["use_std_train_test_split"]:
            indices_dict = collections.OrderedDict()
            indices_dict["gaze_train_idx"] = self.gaze_dataset_indices["train"]
            indices_dict["gaze_test_idx"] = self.gaze_dataset_indices["test"]
            indices_dict["awareness_train_idx"] = self.awareness_dataset_indices["train"]
            indices_dict["awareness_test_idx"] = self.awareness_dataset_indices["test"]
            indices_dict["pairwise_gaze_train_idx"] = self.pairwise_gaze_dataset_indices["train"]
            indices_dict["pairwise_gaze_test_idx"] = self.pairwise_gaze_dataset_indices["test"]

            indices_dict_folder = os.path.join(self.log_dir, "indices_dict_folder")
            os.makedirs(indices_dict_folder, exist_ok=True)
            indices_dict_filename = os.path.join(indices_dict_folder, "indices_dict.pkl")
            with open(indices_dict_filename, "wb") as fp:
                pickle.dump(indices_dict, fp)
