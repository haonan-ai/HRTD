import os
import time
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt
import shapely
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from scipy.special import softmax

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender

from ..post_processing.emergency_brake import EmergencyBrake
from ..post_processing.trajectory_evaluator import TrajectoryEvaluator
from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states, load_checkpoint


class PlutoPlanner(AbstractPlanner):
    requires_scenario: bool = True

    def __init__(
        self,
        planner: TorchModuleWrapper,
        scenario: AbstractScenario = None,
        planner_ckpt: str = None,
        render: bool = False,
        use_gpu=True,
        save_dir=None,
        candidate_subsample_ratio: int = 0.5,
        candidate_min_num: int = 1,
        candidate_max_num: int = 20,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        learning_based_score_weight: float = 0.25,
        use_prediction: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._render = render
        self._imgs = []
        self._scenario = scenario
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self._use_prediction = use_prediction

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
        self._candidate_subsample_ratio = candidate_subsample_ratio
        self._candidate_min_num = candidate_min_num
        self._topk = candidate_max_num

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._scenario_type = scenario.scenario_type

        # post-processing
        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()
        self._learning_based_score_weight = learning_based_score_weight

        if render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        action = self._model_loader.infer(features)
        action = action[0].cpu().numpy()[0]

        return action.astype(np.float64)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)

        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager

        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        planning_trajectory = self._run_planning_once(current_input)

        self._inference_runtimes.append(time.perf_counter() - start_time)

        return planning_trajectory

    def _run_planning_once(self, current_input: PlannerInput):
        ego_state = current_input.history.ego_states[-1]

        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )

        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor()]
        ).to_device(self.device)

        out = self._planner.forward(planner_feature_torch.data)

        # =========================
        # 可视化用：从最终 residual 候选里取出 3 组轨迹，每组若干条
        # 使用已经展平的 flattened_*，避免依赖 coarse_* 细节
        # =========================
        rollout_trajectories = None
        if (
            "flattened_candidate_trajectories" in out
            and "flattened_final_probability" in out
        ):
            flat_traj = (
                out["flattened_candidate_trajectories"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )  # (N, T, 3)
            flat_prob = (
                out["flattened_final_probability"][0].detach().cpu().numpy()
            )  # (N,)

            if flat_traj.ndim == 3 and flat_prob.ndim == 1 and flat_traj.shape[0] == flat_prob.shape[0]:
                N = flat_traj.shape[0]
                group_num = 3
                per_group = 4
                total_needed = group_num * per_group
                if N >= total_needed:
                    top_idx = np.argsort(-flat_prob)[:total_needed]
                elif N >= group_num:
                    # 轨迹不足 12 条时，尽量平均分到 3 组
                    per_group = max(1, N // group_num)
                    total_needed = per_group * group_num
                    top_idx = np.argsort(-flat_prob)[:total_needed]
                else:
                    top_idx = np.argsort(-flat_prob)
                    group_num = N
                    per_group = 1

                groups = []
                for g in range(group_num):
                    start = g * per_group
                    end = min((g + 1) * per_group, len(top_idx))
                    if start >= end:
                        break
                    idx_group = top_idx[start:end]
                    group_local = flat_traj[idx_group]  # (K, T, 3)
                    group_global = self._local_to_global(group_local, ego_state)
                    groups.append(group_global)

                if len(groups) > 0:
                    rollout_trajectories = np.stack(groups, axis=0)  # (G, K, T, 3)

        # 有 residual 时模型输出 5D (R,M,K,T,C)，需用已展平的结果；否则为 4D (R,M,T,C)
        if "flattened_candidate_trajectories" in out and "flattened_final_probability" in out:
            candidate_trajectories = (
                out["flattened_candidate_trajectories"][0].cpu().numpy().astype(np.float64)
            )
            probability = out["flattened_final_probability"][0].cpu().numpy()
        else:
            candidate_trajectories = (
                out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
            )
            probability = out["probability"][0].cpu().numpy()

        if self._use_prediction:
            predictions = out["output_prediction"][0].cpu().numpy()
        else:
            predictions = None

        ref_free_trajectory = (
            (out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in out
            else None
        )

        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            current_input.history.ego_states[-1],
            ref_free_trajectory,
        )

        # 不再进行规则打分，直接使用学习得分选择轨迹
        best_candidate_idx = learning_based_score.argmax()

        # 不再计算规则打分，也就不会更新 TrajectoryEvaluator 的碰撞时间；
        # 这里直接使用正无穷，等价于「无碰撞风险」，保持 EmergencyBrake 接口不变。
        time_to_collision = np.inf
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state,
            time_to_collision,
            candidate_trajectories[best_candidate_idx],
        )

        # no emergency
        if trajectory is None:
            trajectory = candidate_trajectories[best_candidate_idx, 1:]
            trajectory = InterpolatedTrajectory(
                global_trajectory_to_states(
                    global_trajectory=trajectory,
                    ego_history=current_input.history.ego_states,
                    future_horizon=len(trajectory) * self._step_interval,
                    step_interval=self._step_interval,
                    include_ego_state=False,
                )
            )

        if self._render:
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    # 只可视化红/绿/蓝三组 residual 轨迹，不再单独画最终选择的轨迹
                    planning_trajectory=None,
                    # 将全局轨迹转换到以 ego 为原点的坐标系，和其他绘制元素对齐
                    rollout_trajectories=(
                        self._global_to_local(rollout_trajectories, ego_state)
                        if rollout_trajectories is not None
                        else None
                    ),
                    predictions=predictions,
                    return_img=True,
                )
            )

        return trajectory

    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state: EgoState,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        candidate_trajectories: (N, 80, 3) 已展平，或 (n_ref, n_mode, 80, 3) 仅 coarse
        probability: (N,) 或 (n_ref, n_mode)
        """
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
            probability = probability.reshape(-1)

        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    def _get_agent_info(self, data, predictions, ego_state: EgoState):
        """
        predictions: (n_agent, 80, 2 or 3)
        """
        current_velocity = np.linalg.norm(data["agent"]["velocity"][1:, -1], axis=-1)
        current_state = np.concatenate(
            [data["agent"]["position"][1:, -1], data["agent"]["heading"][1:, -1, None]],
            axis=-1,
        )
        velocity = None

        if predictions is None:  # constant velocity
            timesteps = np.linspace(0.1, 8, 80).reshape(1, 80, 1)
            displacement = data["agent"]["velocity"][1:, None, -1] * timesteps
            positions = current_state[:, None, :2] + displacement
            angles = current_state[:, None, 2:3].repeat(80, axis=1)
            predictions = np.concatenate([positions, angles], axis=-1)
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
            velocity = current_velocity[:, None].repeat(81, axis=1)
        elif predictions.shape[-1] == 2:
            predictions = np.concatenate(
                [current_state[:, None, :2], predictions], axis=1
            )
            diff = predictions[:, 1:] - predictions[:, :-1]
            start_end_dist = np.linalg.norm(
                predictions[:, -1, :2] - predictions[:, 0, :2], axis=-1
            )
            near_stop_mask = start_end_dist < 1.0
            angle = np.arctan2(diff[..., 1], diff[..., 0])
            angle = np.concatenate([current_state[:, None, -1], angle], axis=1)
            angle = np.where(
                near_stop_mask[:, None], current_state[:, 2:3].repeat(81, axis=1), angle
            )
            predictions = np.concatenate(
                [predictions[..., :2], angle[..., None]], axis=-1
            )
        elif predictions.shape[-1] == 3:
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
        elif predictions.shape[-1] == 5:
            velocity = np.linalg.norm(predictions[..., 3:5], axis=-1)
            predictions = np.concatenate(
                [current_state[:, None], predictions[..., :3]], axis=1
            )
            velocity = np.concatenate([current_velocity[:, None], velocity], axis=-1)
        else:
            raise ValueError("Invalid prediction shape")

        # to global
        predictions_global = self._local_to_global(predictions, ego_state)

        if velocity is None:
            velocity = (
                np.linalg.norm(np.diff(predictions_global[..., :2], axis=-2), axis=-1)
                / 0.1
            )
            velocity = np.concatenate([current_velocity[..., None], velocity], axis=-1)

        return {
            "tokens": data["agent_tokens"][1:],
            "shape": data["agent"]["shape"][1:, -1],
            "category": data["agent"]["category"][1:],
            "velocity": velocity,
            "predictions": predictions_global,
        }

    def _get_ego_baseline_path(self, reference_lines, ego_state: EgoState):
        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    def _local_to_global(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio

            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report
