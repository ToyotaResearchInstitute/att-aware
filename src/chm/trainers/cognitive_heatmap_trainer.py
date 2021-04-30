import torch
import tqdm


class CHMTrainer(object):
    def __init__(self, params_dict):
        self.params_dict = params_dict

        self.overall_batch_num = 0
        self.max_epochs = self.params_dict.get("max_epochs", 50)
        self.lr_update_num = self.params_dict.get("lr_update_num", 1000)
        self.max_overall_batch_during_training = self.params_dict.get("max_overall_batch_during_training", None)
        self.save_interval = self.params_dict.get("save_interval", 1000)
        self.checkpoint_frequency = self.params_dict.get("checkpoint_frequency", 800)
        self.batch_aggregation_size = self.params_dict.get("batch_aggregation_size", 8)

    def fit(self, module):
        module.trainer = self

        optimizer, scheduler, grad_scaler = module.configure_optimizers()
        gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders = module.get_dataloaders()
        self.is_training_done = False
        for epoch in range(self.max_epochs):
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
    ):
        module.train()
        dataloader_tqdm = tqdm.tqdm(
            enumerate(
                zip(gaze_dataloaders["train"], awareness_dataloaders["train"], pairwise_gaze_dataloaders["train"])
            ),
            desc="train",
        )
        for i, data_batch in dataloader_tqdm:  # iterate through batches
            if (self.overall_batch_num + 1) % self.lr_update_num == 0 and optimizer.param_groups[0][
                "lr"
            ] > self.lr_min_bound:
                print("Update lr")
                scheduler.step()

            if self.max_overall_batch_during_training is not None:
                if self.overall_batch_num > self.max_overall_batch_during_training:
                    self.is_training_done = True
                    break

            self.overall_batch_num += 1

            # save model occasionally
            if (self.overall_batch_num % self.save_interval == 0) and not self.params_dict["no_save_model"]:
                module.save_model(self.overall_batch_num)

            # Test phase during training.
            if (self.overall_batch_num % self.checkpoint_frequency == 1) and not self.params["no_run_test"]:
                self.test(
                    gaze_dataloaders,
                    awareness_dataloaders,
                    pairwise_gaze_dataloaders,
                    module,
                )

            # Training step data_batch is a tuple consisting of (gaze_item, awareness_item, pairwise_gaze_item)
            output = module.training_step(data_batch, self.overall_batch_num)

            # Perform back prop
            loss = output["loss"]
            stats = output["stats"]
            self.cumulative_batch_loss += loss.sum().cpu().detach()
            loss = loss.mean()
            if self.overall_batch_num % self.batch_aggregation_size == 0:
                optimizer.zero_grad()

            grad_scaler.scale(loss).backward()
            if self.overall_batch_num % self.batch_aggregation_size == (self.batch_aggregation_size - 1):
                grad_scaler.step(optimizer)
                grad_scaler.update()

            loss = loss.detach()
            # check for nan loss
            if torch.isnan(loss).sum() or loss.numel() == 0:
                print("skipped: " + str(loss))
            else:
                dataloader_tqdm.set_description("train" + ": {}".format(self.cumulative_batch_loss.detach() / (i + 1)))

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
        assert "driver_facing" in self.params["dropout_ratio"]
        if self.params["dropout_ratio"]["driver_facing"] < (1 - 5e-2):
            module.train(False)
        else:
            module.train(True)

        dataloader_tqdm = tqdm.tqdm(
            enumerate(zip(gaze_dataloaders["test"], awareness_dataloaders["test"])),
            desc="test",
        )
        force_value_strs = ["with_dropout", "with_no_dropout"]
        for j, data_batch in dataloader_tqdm:
            # Testing step data_batch is a tuple consisting of (gaze_item, awareness_item)
            for force_value_str in force_value_strs:  # do one pass each for with and without dropout
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

            import IPython

            IPython.embed(banner1="check test loop")

            self.overall_batch_num += 1

        # set model back to train mode
        module.train(True)