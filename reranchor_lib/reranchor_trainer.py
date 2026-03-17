from functools import partial
from typing import Optional

import datasets
import torch
from datasets import DatasetDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker
from colpali_engine.data.sampler import SingleDatasetBatchSampler
import torch.nn.functional as F
import os
from torch.nn import BCEWithLogitsLoss

class RerAnchorTrainer(Trainer):
    def __init__(self, processor, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
        elif isinstance(kwargs["train_dataset"], list):
            dataset_list = kwargs["train_dataset"]
        else:
            dataset_list = None

        if isinstance(dataset_list, list):
            # round down each dataset if not divible by global batch size
            batch_size = kwargs["args"].train_batch_size
            for i in range(len(dataset_list)):
                if len(dataset_list[i]) % batch_size != 0:
                    total_samples = (len(dataset_list[i]) // batch_size) * batch_size
                    dataset_list[i] = dataset_list[i].take(total_samples)

        if dataset_list is not None:
            kwargs["train_dataset"] = ConcatDataset(dataset_list)

        super().__init__(*args, **kwargs)
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = dataset_list
        self.processor = processor

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataset = self.train_dataset
        description = "Training"
        batch_size = self._train_batch_size
        sampler_fn = self._get_train_sampler
        is_training = True
        dataloader_key = None

        if self.dataset_list is None:
            return super()._get_dataloader(dataset, description, batch_size, sampler_fn, is_training, dataloader_key)

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                ###### batch_sampler set instead of sampler in trainer code #######
                dataloader_params["batch_sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.dataset_list is None:
            return super()._get_train_sampler(train_dataset=train_dataset)

        # Use SingleDatasetBatchSampler to ensure that each dataset in the list is sampled independently
        # Note: Surely breaks in distributed training
        # TODO: fix this
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )
    
    def focal_loss(self, inputs, targets, alpha=0.75, gamma=3.0):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        ret = alpha * (1 - pt) ** gamma * bce_loss
        return ret.mean()

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        outputs = model(**inputs).squeeze(-1)  # (batch_size, sequence_length)
        grount_truth = inputs['selected_tokens']
        loss = []
        for batch_idx in range(len(inputs['input_ids'])):
            mask = grount_truth[batch_idx] != -100
            loss.append(self.focal_loss(outputs[batch_idx][mask], grount_truth[batch_idx][mask]))
        return torch.stack(loss).sum()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")
        with torch.no_grad():
            outputs = model(**inputs).squeeze(-1)  # (batch_size, sequence_length)
            grount_truth = inputs['selected_tokens']
            loss = []
            for batch_idx in range(len(inputs['input_ids'])):
                mask = grount_truth[batch_idx] != -100
                loss.append(self.focal_loss(outputs[batch_idx][mask], grount_truth[batch_idx][mask]))
            return torch.stack(loss).sum(), None, None


class RerAnchorContrastiveTrainer(Trainer):
    def __init__(self, processor, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
        elif isinstance(kwargs["train_dataset"], list):
            dataset_list = kwargs["train_dataset"]
        else:
            dataset_list = None

        if isinstance(dataset_list, list):
            # round down each dataset if not divible by global batch size
            batch_size = kwargs["args"].train_batch_size
            for i in range(len(dataset_list)):
                if len(dataset_list[i]) % batch_size != 0:
                    total_samples = (len(dataset_list[i]) // batch_size) * batch_size
                    dataset_list[i] = dataset_list[i].take(total_samples)

        if dataset_list is not None:
            kwargs["train_dataset"] = ConcatDataset(dataset_list)

        super().__init__(*args, **kwargs)
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = dataset_list
        self.processor = processor
        self.sigmoid_fn = torch.nn.Sigmoid()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataset = self.train_dataset
        description = "Training"
        batch_size = self._train_batch_size
        sampler_fn = self._get_train_sampler
        is_training = True
        dataloader_key = None

        if self.dataset_list is None:
            return super()._get_dataloader(dataset, description, batch_size, sampler_fn, is_training, dataloader_key)

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                ###### batch_sampler set instead of sampler in trainer code #######
                dataloader_params["batch_sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.dataset_list is None:
            return super()._get_train_sampler(train_dataset=train_dataset)
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )
    
    def focal_loss(self, inputs, targets, alpha=0.75, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("instance_")}).squeeze(-1)  # (batch_size, sequence_length)
        grount_truth = inputs['selected_tokens']
        image_losses = []
        final_losses = []
        for batch_idx in range(len(inputs['instance_input_ids'])):
            img_mask = inputs['instance_input_ids'][batch_idx] == 77091
            final_mask =  inputs['instance_input_ids'][batch_idx] == 151655
            image_loss = self.focal_loss(outputs[batch_idx][img_mask], grount_truth[batch_idx][img_mask])
            final_loss = self.focal_loss(outputs[batch_idx][final_mask], grount_truth[batch_idx][final_mask])
            image_losses.append(image_loss)
            final_losses.append(final_loss)

        avg_img_losses = torch.stack(image_losses).mean()
        avg_final_losses = torch.stack(final_losses).mean()
        print(f"Avg Image Loss: {avg_img_losses.item()}, Avg Final Score Loss: {avg_final_losses.item()}")
        return avg_img_losses + avg_final_losses

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")
        with torch.no_grad():
            outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("instance_")}).squeeze(-1)  # (batch_size, sequence_length)
            grount_truth = inputs['selected_tokens']
            image_losses = []
            final_losses = []
            for batch_idx in range(len(inputs['instance_input_ids'])):
                img_mask = inputs['instance_input_ids'][batch_idx] == 77091
                final_mask =  inputs['instance_input_ids'][batch_idx] == 151655
                image_loss = self.focal_loss(outputs[batch_idx][img_mask], grount_truth[batch_idx][img_mask])
                final_loss = self.focal_loss(outputs[batch_idx][final_mask], grount_truth[batch_idx][final_mask])
                image_losses.append(image_loss)
                final_losses.append(final_loss)

            avg_img_losses = torch.stack(image_losses).mean()
            avg_final_losses = torch.stack(final_losses).mean()
            print(f"Avg Image Loss: {avg_img_losses.item()}, Avg Final Score Loss: {avg_final_losses.item()}")
            return avg_img_losses + avg_final_losses, None, None
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override _save to include custom classifier head"""
        # Save the main model using parent's method
        super()._save(output_dir, state_dict)
        
        # Save custom layers separately
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract and save classifier head
        custom_state = {
            'classifier': self.model.classifier.state_dict(),
            'dropout': self.model.dropout.state_dict()
        }
        
        # Handle wrapped models (DDP, FSDP, etc.)
        if hasattr(self.model, 'module'):
            custom_state = {
                'classifier': self.model.module.classifier.state_dict(),
                'dropout': self.model.module.dropout.state_dict()
            }
        
        classifier_path = os.path.join(output_dir, 'classifier_head.pt')
        torch.save(custom_state, classifier_path)
