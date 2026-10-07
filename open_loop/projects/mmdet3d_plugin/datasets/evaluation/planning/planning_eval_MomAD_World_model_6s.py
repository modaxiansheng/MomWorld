from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon
from pyquaternion import Quaternion

from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from nuscenes.nuscenes import NuScenes
from .planning_metric_summary import (
    cumulative_metric_summary, validate_metric_values, validate_valid_samples,
)


def get_lidar_to_global(info):
    lidar_to_ego = np.eye(4)
    lidar_to_ego[:3, :3] = Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar_to_ego[:3, 3] = np.asarray(info["lidar2ego_translation"])
    ego_to_global = np.eye(4)
    ego_to_global[:3, :3] = Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego_to_global[:3, 3] = np.asarray(info["ego2global_translation"])
    return ego_to_global @ lidar_to_ego


def align_previous_plan(previous_plan, previous_info, current_info):
    """Move the previous prediction into the current lidar frame and time."""
    previous_to_current = (
        np.linalg.inv(get_lidar_to_global(current_info))
        @ get_lidar_to_global(previous_info)
    )
    transform = previous_plan.new_tensor(previous_to_current)
    points = previous_plan.new_zeros(
        previous_plan.shape[0], previous_plan.shape[1], 4
    )
    points[..., :2] = previous_plan
    points[..., 3] = 1
    aligned = torch.matmul(points, transform.transpose(0, 1))[..., :2]
    # Current t+0.5 corresponds to previous t+1.0. Repeat only the last point,
    # for which no t+6.5 prediction exists.
    return torch.cat([aligned[:, 1:], aligned[:, -1:]], dim=1)

def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if  boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly =  Polygon([(point[0], point[1]) for point in corners_box[i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True

    return False

def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0]+1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]

class PlanningMetric():
    def __init__(
        self,
        n_future=12,
        compute_on_step: bool = False,
    ):
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.Consist = torch.zeros(self.n_future)
        self.total = torch.tensor(0)
        self.consist_total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        for t in range(n_future):
            if t >= len(fut_boxes):
                continue
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only supprt bs=1'
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            
            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def compute_Consist(self, trajs, last_final_planning,gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''

        return torch.sqrt(((((trajs[:, :, :2] - last_final_planning[:, :, :2])) ** 2) * gt_trajs_mask).sum(dim=-1)) 
    def update(
        self,
        trajs,
        gt_trajs,
        gt_trajs_mask,
        fut_boxes,
        last_final_planning,
        consistency_valid,
    ):
        """
        trajs  pred_final_planning
        gt_trajs  gt_ego_fut_trajs
        """
        # import pdb; pdb.set_trace()
        assert trajs.shape == gt_trajs.shape
        # Work on local clones: the baseline evaluator changed result tensors
        # in place, which could affect later metrics using the same results.
        trajs = trajs.clone()
        gt_trajs = gt_trajs.clone()
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        Consist = self.compute_Consist(
            trajs, last_final_planning, gt_trajs, gt_trajs_mask
        )
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], fut_boxes)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        if consistency_valid:
            self.Consist += Consist.sum(dim=0)
            self.consist_total += len(trajs)
        self.total +=len(trajs)

    def compute(self):
        validate_valid_samples(self.total.item())
        denominator = self.total.clamp(min=1)
        return {
            'obj_col': self.obj_col / denominator,
            'obj_box_col': self.obj_box_col / denominator,
            'L2' : self.L2 / denominator,
            'Consist': self.Consist / self.consist_total.clamp(min=1)
        }

# nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuscenes/", verbose=True)
def planning_eval_MomAD_World_model_6s(results, eval_config, logger):
    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)
    n_future = 12
    planning_metrics = PlanningMetric(n_future=n_future)
    last_final_planning = torch.zeros([1, n_future, 2])
    last_scene_token = None
    last_info = None
    # import pdb; pdb.set_trace()

    for i, data in enumerate(tqdm(dataloader)):
        # next_token=nusc.get('sample', data['img_metas'].data[0][0]["token"])["next"]
        # import pdb; pdb.set_trace()
        sdc_planning = data['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        sdc_planning_mask = data['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        fut_boxes = data['fut_boxes']
        if not sdc_planning_mask.all(): ## for incomplete gt, we do not count this sample
            continue
        scene_token = dataset.data_infos[i]['scene_token']
        if scene_token != last_scene_token:
            last_final_planning.zero_()
            last_scene_token = scene_token
            last_info = None
        current_info = dataset.data_infos[i]
        consistency_valid = last_info is not None
        if consistency_valid:
            aligned_last_planning = align_previous_plan(
                last_final_planning, last_info, current_info
            )
        else:
            aligned_last_planning = last_final_planning
        res = results[i]
        pred_sdc_traj = res['img_bbox']['final_planning'].unsqueeze(0)
        planning_metrics.update(
            pred_sdc_traj[:, :n_future, :2],
            sdc_planning[0, :, :n_future, :2],
            sdc_planning_mask[0, :, :n_future, :2],
            fut_boxes,
            aligned_last_planning,
            consistency_valid,
        )
        last_final_planning = pred_sdc_traj[:, :n_future, :2].clone()
        last_info = current_info
    valid_samples = planning_metrics.total.item()
    consistency_valid_samples = planning_metrics.consist_total.item()
    planning_results = planning_metrics.compute()
    planning_metrics.reset()
    from prettytable import PrettyTable
    planning_tab = PrettyTable()
    metric_dict = {}

    planning_tab.field_names = ["metrics"] + [
        "%.1fs" % ((i + 1) * 0.5) for i in range(n_future)
    ] + ["avg@1,2,3s"]
    for key in planning_results.keys():
        metric_valid_samples = consistency_valid_samples if key == 'Consist' else valid_samples
        validate_metric_values(planning_results[key].tolist(), metric_valid_samples)
        raw_value = np.asarray(planning_results[key].tolist(), dtype=np.float64)
        cumulative_value = np.cumsum(raw_value) / np.arange(1, n_future + 1)
        short_avg = float(np.mean(cumulative_value[[1, 3, 5]]))
        long_avg = float(np.mean(cumulative_value[[7, 9, 11]]))
        exact_6s = float(raw_value[11])
        display_value = cumulative_value.tolist() + [short_avg]
        # Keep the original key for fair 6s-baseline comparison, and expose
        # explicit long-horizon metrics required by this experiment.
        metric_dict[key] = short_avg
        cumulative_name = 'ADE' if key in ('L2', 'Consist') else 'cumulative'
        metric_dict[key + '_' + cumulative_name + '_4s_5s_6s'] = long_avg
        metric_dict[key + '_at_6s'] = exact_6s
        metric_dict.update(cumulative_metric_summary(
            key, cumulative_value.tolist(), metric_valid_samples,
        ))
        # import pdb; pdb.set_trace()
        row_value = []
        row_value.append(key)
        for i in range(len(display_value)):
            if 'col' in key:
                row_value.append('%.3f' % float(display_value[i]*100) + '%')
            else:
                row_value.append('%.4f' % float(display_value[i]))
        planning_tab.add_row(row_value)

    print_log('\n'+str(planning_tab), logger=logger)
    return metric_dict
