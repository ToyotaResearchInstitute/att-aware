# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import tqdm


class CHMTrainer(object):
    """
    Trainer class that is responsible for the main training loop. Includes the testing as well as visualization phase
    """

    def __init__(self, params_dict):
        """
        Parameters:
        ----------
        params_dict: dict
            params dict containing the args from args_file.py
        """
        self.params_dict = params_dict

        # counter for training
        self.overall_batch_num = 0
        self.overall_training_batch_num = 0
        # Maximum number of epochs to train
        self.max_epochs = self.params_dict.get("max_epochs", 50)
        # Interval at which the learning rate is updated
        self.lr_update_num = self.params_dict.get("lr_update_num", 1000)
        # Lower bound for learning rate
        self.lr_min_bound = self.params_dict.get("lr_min_bound", 1e-4)
        # If not None, indicates the maximum number of batches used for training. Supersedes self.max_epochs
        self.max_overall_batch_during_training = self.params_dict.get("max_overall_batch_during_training", None)
        # Interval at which the model is saved to disk
        self.save_interval = self.params_dict.get("save_interval", 1000)
        # Interval at which the testing phase is activated
        self.checkpoint_frequency = self.params_dict.get("checkpoint_frequency", 800)
        # Interval at which the visualization to tensorboard is activated
        self.visualize_frequency = self.params_dict.get("visualize_frequency", 250)
        # Number of examples to visualize
        self.num_visualization_examples = self.params_dict.get("num_visualization_examples", 5)
        # Number of times (batches) the gradients are accumulated before back prop happens
        self.batch_aggregation_size = self.params_dict.get("batch_aggregation_size", 8)
        # Strings indicating whether the side channel is "with" or "without" gaze
        self.force_value_strs = ["with_gaze", "without_gaze"]

    def fit(self, module, ds_type="train"):
        module.trainer = self

        # configure optimizer, lr scheduler and gradient scaler.
        optimizer, scheduler, grad_scaler = module.configure_optimizers()
        # retrieve dataloaders
        gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders = module.get_dataloaders()
        self.is_training_done = False
        # start outer training loop
        for epoch in range(self.max_epochs):
            self.epoch_num = epoch
            print("Epoch number ", self.epoch_num)
            if self.is_training_done:
                break
            self.cumulative_batch_loss = torch.tensor(0.0)
            self.train(
                gaze_dataloaders,
                awareness_dataloaders,
                pairwise_gaze_dataloaders,
                module,
                optimizer,
                scheduler,
                grad_scaler,
                ds_type=ds_type,
            )

    def train(
        self,
        gaze_dataloaders,
        awareness_dataloaders,
        pairwise_gaze_dataloaders,
        module,
        optimizer,
        scheduler,
        grad_scaler,
        ds_type="train",
    ):
        module.train()

        assert ds_type == "train" or ds_type == "test", "The dataset type has to be either train or test"
        if ds_type == "train":
            print("Using train sets for training")
            dataloader_tqdm = tqdm.tqdm(
                enumerate(
                    zip(gaze_dataloaders["train"], awareness_dataloaders["train"], pairwise_gaze_dataloaders["train"])
                ),
                desc="train",
            )
            description = "train"
        elif ds_type == "test":
            """
            This is used for calibration experiment, in which the training is done on the test set
            """
            print("Using test sets for training")
            dataloader_tqdm = tqdm.tqdm(
                enumerate(
                    zip(gaze_dataloaders["test"], awareness_dataloaders["test"], pairwise_gaze_dataloaders["test"])
                ),
                desc="train_using_test",
            )
            description = "train_using_test"

        for training_batch_i, data_batch in dataloader_tqdm:  # iterate through batches
            if (training_batch_i + 1) % self.lr_update_num == 0 and optimizer.param_groups[0]["lr"] > self.lr_min_bound:
                print("Update lr")
                scheduler.step()

            if self.max_overall_batch_during_training is not None:
                if training_batch_i > self.max_overall_batch_during_training:
                    self.is_training_done = True
                    break

            # increments during training and testing loop, used for proper tensorboard logging
            self.overall_batch_num += 1
            # only increments for the training
            self.overall_training_batch_num += 1
            # visualize output occasionally
            if (training_batch_i + 1) % self.visualize_frequency == 0:
                self.visualize(gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders, module)

            # save model occasionally
            if ((training_batch_i + 1) % self.save_interval == 0) and not self.params_dict["no_save_model"]:
                module.save_model(self.overall_training_batch_num)

            # Test phase during training.
            if ((training_batch_i + 1) % self.checkpoint_frequency == 0) and not self.params_dict["no_run_test"]:
                self.test(gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders, module)

            # Training step. data_batch is a tuple consisting of (gaze_item, awareness_item, pairwise_gaze_item)
            output = module.training_step(data_batch, self.overall_batch_num)

            # Perform back prop
            loss = output["loss"]
            stats = output["stats"]
            self.cumulative_batch_loss += loss.sum().cpu().detach()
            loss = loss.mean()
            if training_batch_i % self.batch_aggregation_size == 0:
                optimizer.zero_grad()

            grad_scaler.scale(loss).backward()
            if training_batch_i % self.batch_aggregation_size == (self.batch_aggregation_size - 1):
                grad_scaler.step(optimizer)
                grad_scaler.update()

            loss = loss.detach()
            # check for nan loss
            if torch.isnan(loss).sum() or loss.numel() == 0:
                print("skipped: " + str(loss))
            else:
                dataloader_tqdm.set_description(
                    description + ": {}".format(self.cumulative_batch_loss.detach() / (training_batch_i + 1))
                )

            if module.logger is not None:
                module.logger.add_scalar(
                    tag="train" + "/batch_loss", scalar_value=loss.cpu().detach(), global_step=self.overall_batch_num
                )
                for key in stats.keys():
                    if type(stats[key]) is dict:
                        continue
                    else:
                        module.logger.add_scalar(
                            tag="train" + "/batch_" + key,
                            scalar_value=stats[key].cpu().detach(),
                            global_step=self.overall_batch_num,
                        )

            del loss

    def test(self, gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders, module):
        # set model to eval mode
        assert "driver_facing" in self.params_dict["dropout_ratio"]
        if self.params_dict["dropout_ratio"]["driver_facing"] < (1 - 5e-2):
            module.train(False)
        else:
            module.train(True)

        # create data loader for gaze and awareness
        dataloader_tqdm = tqdm.tqdm(
            enumerate(zip(gaze_dataloaders["test"], awareness_dataloaders["test"], pairwise_gaze_dataloaders["test"])),
            desc="test",
        )

        for j, data_batch in dataloader_tqdm:
            # Testing step data_batch is a tuple consisting of (gaze_item, awareness_item)
            for force_value_str in self.force_value_strs:  # do one pass each for with and without dropout
                output = module.testing_step(data_batch, self.overall_batch_num, force_value_str)
                loss = output["loss"]
                stats = output["stats"]
                loss = loss.mean()
                loss = loss.detach()
                if module.logger is not None:
                    module.logger.add_scalar(
                        tag="test" + "/batch_loss_" + force_value_str,
                        scalar_value=loss.cpu().detach(),
                        global_step=self.overall_batch_num,
                    )
                    for key in stats.keys():
                        if type(stats[key]) is dict:
                            continue
                        else:
                            module.logger.add_scalar(
                                tag="test" + "/batch_" + key + "_" + force_value_str,
                                scalar_value=stats[key].cpu().detach(),
                                global_step=self.overall_batch_num,
                            )

                del loss

            self.overall_batch_num += 1

        # set model back to train mode
        module.train(True)

    def visualize(self, gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders, module):
        if module.logger is not None:
            assert "driver_facing" in self.params_dict["dropout_ratio"]
            if self.params_dict["dropout_ratio"]["driver_facing"] < (1 - 5e-2):
                module.train(False)
            else:
                module.train(True)

            # visualize testing data create data loader for gaze and awareness
            dataloader_tqdm = tqdm.tqdm(
                enumerate(zip(gaze_dataloaders["test"], awareness_dataloaders["test"])),
                desc="visualize",
            )
            for j, data_batch in dataloader_tqdm:
                # Visualization step data_batch is a tuple consisting of (gaze_item, awareness_item)
                for force_value_str in self.force_value_strs:  # do one pass each for with and without dropout
                    print("VISUALIZATION ", force_value_str)
                    output = module.visualization_step(
                        data_batch, self.overall_batch_num, force_value_str, self.num_visualization_examples
                    )

                break

            # after visualization put the model back in training mode.
            module.train(True)