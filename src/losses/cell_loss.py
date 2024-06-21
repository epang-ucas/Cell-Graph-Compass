import logging
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.utils import tensor_tree_map
from unicore.losses import UnicoreLoss, register_loss
from unicore.data import data_utils

import numpy as np

from torch.profiler import profile
import torchmetrics
import pickle


@register_loss("single_cell_loss")
class ScLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.criterion = self.masked_mse_loss_sum
        self.criterion_mvc = F.mse_loss
        self.criterion_dab = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion_bernoulli = self.criterion_neg_log_bernoulli_sum
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction="sum")
    
    def masked_mse_loss_sum(
        self,
        input: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the masked MSE loss between input and target.
        """
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss
    
    def criterion_neg_log_bernoulli_sum(
        self,
        input: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood of Bernoulli distribution
        """
        mask = mask.float()
        bernoulli = torch.distributions.Bernoulli(probs=input)
        masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
        return -masked_log_probs.sum()

    def forward(self, model, batch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        outputs = model(batch)

        loss, sample_size, logging_output = self.loss(outputs, batch, model)
        return loss, sample_size, logging_output

    def loss(self, outputs, batch, model):
        input_values = batch["values"]
        target_values = batch["truth"]
        # target_values = batch["truth"] / (model.n_bins - 1)
        batch_labels = batch["batch_id"]
        # input_gene_ids = batch["gene_list"]
        celltype_ids = batch['celltype']
        bsz = input_values.shape[0]
        logging_output =dict()
        loss = 0.0
        type_ = target_values.dtype
        masked_positions = outputs["mask_position"]
        mask_size = int(masked_positions.sum().item())
        logging_output["mask_size"] = mask_size
        if model.do_mvg:
            loss_mse = self.criterion(
                outputs["mlm_output"].to(type_), target_values, masked_positions
            )/mask_size*bsz
            loss = loss_mse + loss
            logging_output["train/mse"] = loss_mse.item()
        if model.explicit_zero_prob:
            loss_zero_log_prob = self.criterion_bernoulli(
                outputs["mlm_zero_probs"].to(type_), target_values, masked_positions
            )
            loss = loss_zero_log_prob + loss
            logging_output["train/nzlp"] = loss_zero_log_prob.item()
        if model.do_mvc:
            loss_mvc = self.criterion_mvc(
                outputs["mvc_output"].to(type_), 
                target_values,
                reduction = "mean"
            )*bsz
            loss = loss_mvc + loss
            logging_output["train/mvc"] = loss_mvc.item()
        if model.do_mvc and model.explicit_zero_prob:
            loss_gepc_zero_log_prob = self.criterion_bernoulli(
                outputs["mvc_zero_probs"].to(type_), target_values, masked_positions
            )
            loss = loss_gepc_zero_log_prob + loss
            logging_output["train/mvc_nzlp"] = loss_gepc_zero_log_prob.item()
        if model.do_ecs:
            loss_ecs = outputs["loss_ecs"].to(type_)
            loss_ecs = loss_ecs * bsz * 10
            loss = loss_ecs + loss
            logging_output["train/ecs"] = loss_ecs.item()
        if model.do_dab:
            loss_dab = self.criterion_dab(
                outputs["dab_output"].to(type_), batch_labels.long()
            )
            loss = model.dab_weight * loss_dab + loss
            logging_output["train/dab"] = loss_dab.item()
        if model.do_cls:
            loss_cls = self.criterion_cls(
                outputs["cls_output"].to(type_), celltype_ids.long()
            )
            loss_cls = loss_cls * 40
            loss = loss_cls + loss
            # loss = model.cls_weight * loss_cls + loss
            logging_output["train/cls"] = loss_cls.item()
            acc_macro = torchmetrics.functional.accuracy(
                outputs["cls_output"], 
                celltype_ids, 
                task = "multiclass", 
                num_classes = model.n_cls, 
                average = 'macro',
            )
            acc_micro = torchmetrics.functional.accuracy(
                outputs["cls_output"], 
                celltype_ids, 
                task = "multiclass", 
                num_classes = model.n_cls, 
                average = 'micro',
            )
            logging_output["train/acc_macro"] = acc_macro.item() * bsz
            logging_output["train/acc_micro"] = acc_micro.item() * bsz
            
        loss = loss.to(model.dtype)
        logging_output["loss"] = loss.item()
        logging_output["bsz"] = bsz
        logging_output["sample_size"] = bsz 

        return loss, bsz, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        for key in logging_outputs[0]:
            if key in ["sample_size", "mask_size", "bsz", "loss"]:
                continue
            else:
                value_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, value_sum / sample_size, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
