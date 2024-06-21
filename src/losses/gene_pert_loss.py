import logging
import torch
import torch.nn.functional as F
import torchvision
from unicore import metrics
from unicore.utils import tensor_tree_map
from unicore.losses import UnicoreLoss, register_loss
from unicore.data import data_utils

import numpy as np

from torch.profiler import profile
import torchmetrics



@register_loss("pert_loss")
class PertLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.criterion = F.mse_loss
        self.criterion_dab = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion_bernoulli = self.criterion_neg_log_bernoulli
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def masked_mse_loss(
        self,
        input: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the masked MSE loss between input and target.
        """
        if mask:
            mask = mask.float()
            loss = F.mse_loss(input * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        else:
            loss = F.mse_loss(input, target, reduction="sum")
        return loss
    
    def criterion_neg_log_bernoulli(
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
        return -masked_log_probs.sum() / mask.sum()

    def forward(self, model, batch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb;pdb.set_trace()
        outputs = model(batch)

        loss, sample_size, logging_output = self.loss(outputs, batch, model)
        return loss, sample_size, logging_output

    def loss(self, outputs, batch, model):
        input_values = batch["values"]
        target_values = batch["target"]
        input_vocab_ids = batch["gene_list"]
        type_ = target_values.dtype
        bsz = input_values.shape[0]
        logging_output =dict()
        loss = 0.0
        src_key_padding_mask = input_vocab_ids.eq(model.vocab[model.pad_token])
        # import pdb;pdb.set_trace()
        assert src_key_padding_mask.any() == False
        assert model.do_mvg
        if model.do_mvg:
            target_positions = ~src_key_padding_mask
            if model.mvg_decoder_style == 'continuous':
                loss_mvg = self.criterion(
                    outputs["mlm_output"].to(type_) * target_positions, 
                    target_values * target_positions,
                    reduction = "sum"
                ) / target_positions.sum()
                loss_mvg = loss_mvg * bsz
                loss = loss_mvg + loss
                logging_output["train/loss_mvg"] = loss_mvg.item()
                if model.explicit_zero_prob:
                    loss_zero_log_prob = self.criterion_bernoulli(
                        outputs["mlm_zero_probs"].to(type_), target_values, target_positions
                    )
                    loss_zero_log_prob = loss_zero_log_prob * bsz
                    loss = loss_zero_log_prob + loss
                    logging_output["train/nzlp"] = loss_zero_log_prob.item()
            elif model.mvg_decoder_style == 'category':
                # import pdb;pdb.set_trace()
                truths_delta = target_values - input_values
                truth_change = torch.where(torch.abs(truths_delta) < np.log(2), 0, 1).long()
                up_logits = outputs["up_logits"]
                change_logits = outputs["change_logits"]

                loss_change = torchvision.ops.sigmoid_focal_loss(
                    inputs = change_logits.view(-1)[target_positions.view(-1).bool()].to(type_), 
                    targets = truth_change.view(-1)[target_positions.view(-1).bool()].to(type_),
                    alpha = model.focal_alpha, 
                    gamma = model.focal_gamma, 
                    reduction = 'sum'
                ) / target_positions.sum()
                loss_change = loss_change / (model.focal_alpha)
                loss_change = loss_change * bsz * model.focal_weight
                loss = loss + loss_change
                logging_output["train/loss_change"] = loss_change.item()

                pred_change = torch.sigmoid(change_logits) >= 0.5
                acc_change = torch.sum((pred_change == truth_change) * target_positions) 
                logging_output["train/acc_change"] = acc_change.item()
                logging_output["target_positions_size"] = target_positions.sum()
                if model.five_class:
                    truth_up_5 = truths_delta >= np.log(5)
                    truth_up_2 = (~truth_up_5) * (truths_delta >= np.log(2))
                    truth_down_5 = truths_delta <= -1*np.log(5)
                    truth_down_2 = (~truth_down_5) * (truths_delta <= -1*np.log(2))
                    truth_label = truth_down_2*1 + truth_up_2*2 + truth_up_5*3
                    target_positions = target_positions * truth_change
                    loss_up =  torch.nn.functional.cross_entropy(
                        input=up_logits.reshape(-1,4)[target_positions.view(-1).bool()],
                        target=truth_label.reshape(-1)[target_positions.view(-1).bool()],
                        reduction='sum',
                    ) / target_positions.sum()
                    pred_up = torch.argmax(up_logits, dim=-1)
                    acc_up = torch.sum((pred_up == truth_label) * target_positions).to(type_)
                else:
                    truth_up = (truths_delta > 0).long()
                    truth_up = truth_up * truth_change - 1 * ( 1 - truth_change)
                    target_positions = target_positions * truth_change
                    loss_up = torch.nn.functional.binary_cross_entropy_with_logits(
                        input = up_logits.view(-1)[target_positions.view(-1).bool()].to(type_), 
                        target = truth_up.view(-1)[target_positions.view(-1).bool()].to(type_),
                        reduction = 'sum',
                    ) / target_positions.sum()
                    pred_up = torch.sigmoid(up_logits) >= 0.5
                    acc_up = torch.sum((pred_up == truth_up) * target_positions) 
                loss_up = loss_up * bsz
                loss = loss + loss_up
                logging_output["train/loss_up"] = loss_up.item()
                logging_output["train/acc_up"] = acc_up.item()
                logging_output["change_positions_size"] = target_positions.sum()

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
        target_positions_size = sum(log.get("target_positions_size", 0) for log in logging_outputs)
        change_positions_size = sum(log.get("change_positions_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        for key in logging_outputs[0]:
            if key in ["sample_size", "bsz", "loss", "target_positions_size", "change_positions_size"]:
                continue
            elif key in ["train/acc_change"]:
                value_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, value_sum / target_positions_size, target_positions_size, round=4)
            elif key in ["train/acc_up"]:
                value_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, value_sum / change_positions_size, change_positions_size, round=4)
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
