import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.num_residual_modes = getattr(model, "num_residual_modes", 0)
        self.mode_interval = self.radius / self.num_modes

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, targets, scenarios = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory = res["trajectory"][:train_num]
        probability = res["probability"][:train_num]
        res_trajectory = (
            res["res_trajectory"][:train_num]
            if res.get("res_trajectory", None) is not None
            else None
        )
        res_probability = (
            res["res_probability"][:train_num]
            if res.get("res_probability", None) is not None
            else None
        )
        prediction = res["prediction"][:train_num]

        ref_free_trajectory = res.get("ref_free_trajectory", None)

        targets_pos = data["agent"]["target"][:train_num]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:]
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:]

        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        ego_reg_loss, ego_cls_loss, ego_res_cls_loss, collision_loss = self.get_planning_loss(
            data=data,
            trajectory=trajectory,
            probability=probability,
            res_trajectory=res_trajectory,
            res_probability=res_probability,
            valid_mask=valid_mask[:, 0],
            target=target[:, 0],
            bs=train_num,
        )

        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum().clamp_min(1)
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        )

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + ego_res_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + ego_ref_free_reg_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "res_cls_loss": ego_res_cls_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, T, 6)
        """
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum().clamp_min(1)

        return prediction_loss

    def get_planning_loss(
        self,
        data,
        trajectory,
        probability,
        res_trajectory,
        res_probability,
        valid_mask,
        target,
        bs,
    ):
        """
        Args:
            trajectory: (bs, R, M, T, 6)
            probability: (bs, R, M)
            res_trajectory: (bs, R, M, K, T, 6) or None
            res_probability: (bs, R, M, K) or None
            valid_mask: (bs, T)
            target: (bs, T, 6)
        """
        device = trajectory.device
        batch_idx = torch.arange(bs, device=device)

        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s

        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)

        future_projection = data["reference_line"]["future_projection"][:bs][
            batch_idx, :, endpoint_index
        ]  # (bs, R, 2)

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )  # (bs,)

        target_m_index = (
            future_projection[batch_idx, target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        masked_probability = probability.masked_fill(
            r_padding_mask.unsqueeze(-1), -1e6
        )  # (bs, R, M)

        coarse_label = torch.zeros_like(masked_probability)
        coarse_label[batch_idx, target_r_index, target_m_index] = 1.0

        cls_loss = F.cross_entropy(
            masked_probability.reshape(bs, -1),
            coarse_label.reshape(bs, -1).detach(),
        )

        if res_trajectory is not None and res_probability is not None:
            selected_res_traj = res_trajectory[
                batch_idx, target_r_index, target_m_index
            ]  # (bs, K, T, 6)

            # match best residual branch by trajectory distance
            traj_dist = torch.abs(selected_res_traj - target.unsqueeze(1)).sum(-1)  # (bs, K, T)
            traj_dist = (
                traj_dist * valid_mask.unsqueeze(1)
            ).sum(-1) / valid_mask.unsqueeze(1).sum(-1).clamp_min(1)  # (bs, K)

            target_k_index = traj_dist.argmin(dim=-1)  # (bs,)

            best_trajectory = selected_res_traj[batch_idx, target_k_index]  # (bs, T, 6)

            selected_res_probability = res_probability[
                batch_idx, target_r_index, target_m_index
            ]  # (bs, K)

            res_cls_loss = F.cross_entropy(
                selected_res_probability,
                target_k_index.detach(),
            )
        else:
            best_trajectory = trajectory[batch_idx, target_r_index, target_m_index]
            res_cls_loss = trajectory.new_zeros(1)

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, res_cls_loss, collision_loss

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.
        """
        trajectory = res["trajectory"]
        probability = res["probability"]
        res_trajectory = res.get("res_trajectory", None)
        res_probability = res.get("res_probability", None)

        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)  # (bs, R)
        bs = probability.shape[0]
        batch_idx = torch.arange(bs, device=probability.device)

        if res_trajectory is not None and res_probability is not None:
            masked_probability = probability.masked_fill(
                r_padding_mask.unsqueeze(-1), -1e6
            )  # (bs, R, M)

            masked_res_probability = res_probability.masked_fill(
                r_padding_mask.unsqueeze(-1).unsqueeze(-1), -1e6
            )  # (bs, R, M, K)

            final_probability = masked_probability.unsqueeze(-1) + masked_res_probability
            # (bs, R, M, K)

            bs_, R, M, K, T, _ = res_trajectory.shape
            assert bs_ == bs

            flat_probability = final_probability.reshape(bs, R * M * K)
            flat_trajectory = res_trajectory.reshape(bs, R * M * K, T, -1)

            k = min(6, flat_probability.shape[-1])
            top_k_prob, top_k_index = flat_probability.topk(k, dim=-1)
            top_k_traj = flat_trajectory[batch_idx[:, None], top_k_index]
        else:
            masked_probability = probability.masked_fill(
                r_padding_mask.unsqueeze(-1), -1e6
            )

            bs_, R, M, T, _ = trajectory.shape
            assert bs_ == bs

            flat_probability = masked_probability.reshape(bs, R * M)
            flat_trajectory = trajectory.reshape(bs, R * M, T, -1)

            k = min(6, flat_probability.shape[-1])
            top_k_prob, top_k_index = flat_probability.topk(k, dim=-1)
            top_k_traj = flat_trajectory[batch_idx[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)

        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print("unused param", name)