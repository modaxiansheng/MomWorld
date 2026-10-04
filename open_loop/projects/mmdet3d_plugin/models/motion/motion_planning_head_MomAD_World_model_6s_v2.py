from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.motion.motion_blocks import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk
from .latent_world_model_MomAD_World_model_6s import LatentWorldModelMomAD6s
from .next_token_prediction import NextTokenPredictor


@HEADS.register_module()
class MotionPlanningHead_MomAD_World_model_6s_V2(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=12,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_rescore=True,
        latent_world_model=None,
        world_plan_cls_weight=0.25,
        world_plan_reg_weight=0.5,
        world_scene_flow_weight=0.1,
        world_diversity_weight=0.01,
        world_latent_weight=0.1,
        world_state_reconstruction_weight=0.1,
        long_horizon_weight=2.0,
        final_fusion_bias=-4.595,
        world_fusion_enabled=False,
        eval_planning_branch="final",
        time_adaptive_fusion=False,
        consistency_aware_fusion=False,
    ):
        super(MotionPlanningHead_MomAD_World_model_6s_V2, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_plan_modes = 3 * ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.use_rescore = use_rescore
        self.world_plan_cls_weight = world_plan_cls_weight
        self.world_plan_reg_weight = world_plan_reg_weight
        self.world_scene_flow_weight = world_scene_flow_weight
        self.world_diversity_weight = world_diversity_weight
        self.world_latent_weight = world_latent_weight
        self.world_state_reconstruction_weight = world_state_reconstruction_weight
        self.long_horizon_weight = long_horizon_weight
        self.world_fusion_enabled = world_fusion_enabled
        self.eval_planning_branch = eval_planning_branch
        self.time_adaptive_fusion = time_adaptive_fusion
        self.consistency_aware_fusion = consistency_aware_fusion
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        self.last_op = ""

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map
        self.refine_2th_layer = MotionPlanning2thRefinementModule(
            embed_dims=embed_dims,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
        )
        latent_world_model = latent_world_model or {}
        self.latent_world_model = LatentWorldModelMomAD6s(
            embed_dims=embed_dims,
            future_steps=ego_fut_ts,
            num_plan_modes=self.num_plan_modes,
            **latent_world_model,
        )
        self.final_fusion_bias = nn.Parameter(
            torch.tensor(float(final_fusion_bias))
        )
        self.register_buffer(
            "last_final_planning_prediction",
            torch.zeros(1, self.ego_fut_ts, 2),
            persistent=False,
        )
        # Parameter-only compatibility with the supplied 6s checkpoint.  The
        # legacy module is intentionally not called: its global cache is not
        # safe for shuffled/distributed batches.  Keeping its registered name
        # avoids dropping already-trained checkpoint tensors during loading.
        self.next_token_predictor = NextTokenPredictor(
            embed_dims, embed_dims // 2
        )
        del self.next_token_predictor.fc
        for parameter in self.next_token_predictor.parameters():
            parameter.requires_grad = False


    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        
        for m in self.modules():
            #import pdb;pdb.set_trace()
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar


    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   

        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        #det_output包括 ['clas sification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id']
        #det_output['classification'][0].shape  torch.Size([6, 900, 10]) (cam,anchor,class)
        #det_output['prediction'][0].shape  torch.Size([6, 900, 11]) (cam,anchor,11) 11:{x, y, z, ln w, ln h, ln l, sin yaw, cos yaw, vx, vy, vz}
        #det_output['quality'][0].shape  torch.Size([6, 900, 2]) (cam,anchor,2) 2:centerness,yawness
        #det_output['instance_feature'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim) 
        #det_output['anchor_embed'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim) 
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        det_confidence_selected, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )
        #instance_feature_selected.shape  torch.Size([6, 50, 256])
        map_instance_feature = map_output["instance_feature"]
        #map_output包括dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed'])
        #map_output[classification] torch.Size([6, 100, 3])
        #map_output[prediction] torch.Size([6, 100, 40])
        #map_output[quality] [None, None, None, None, None, None]
        #map_output[instance_feature] torch.Size([6, 100, 256])
        #map_output[anchor_embed] torch.Size([6, 100, 256])

        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        map_confidence_selected, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )
       
        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,#torch.Size([6, 1, 256])
            ego_anchor,#[6, 1, 11]
            temp_instance_feature,#torch.Size([6, 901, 1, 256])
            temp_anchor,#torch.Size([6, 901, 1, 11])
            temp_mask,#torch.Size([6, 901, 1])
        ) = self.instance_queue.get(
            det_output,#dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id'])
            feature_maps,#torch.Size([6, 89760, 256]) torch.Size([6, 4, 2]) torch.Size([6, 4])
            metas,#dict_keys(['img_metas', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status'])
            bs,#6
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)#torch.Size([6, 1, 256])
        temp_anchor_embed = anchor_encoder(temp_anchor)#torch.Size([6, 901, 1, 256])
        temp_instance_feature = temp_instance_feature.flatten(0, 1)#torch.Size([5406, 1, 256])
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)#torch.Size([6, 901, 1, 256])
        temp_mask = temp_mask.flatten(0, 1)#torch.Size([6, 901, 1])

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors) #motion_anchor torch.Size([6, 900, 6, 12, 2]) #torch.Size([6, 900, 10])  torch.Size([6, 900, 11])
        plan_anchor = torch.tile( #torch.Size([6, 3, 6, 12, 2])
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :])) #torch.Size([6, 900, 6, 256])
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) #torch.Size([6, 3, 6, 256])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)#torch.Size([6, 1, 18, 256])

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1) #torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)#torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_classification_refined = []
        planning_prediction_refined = []
        planning_status_refined = []
        for i, op in enumerate(self.operation_order):
            #import pdb;pdb.set_trace()
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                #self.last_op = "temp_gnn"
                instance_feature = self.graph_model(#[5406, 1, 256])
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,#torch.Size([5406, 1, 256])
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),#torch.Size([5406, 1, 256])
                    key_pos=temp_anchor_embed,#torch.Size([5406, 1, 256])
                    key_padding_mask=temp_mask,#torch.Size([5406, 1])
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)#torch.Size([6, 901, 256])
            elif op == "gnn":
                #self.last_op = "gnn"
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                #self.last_op = "norm"
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                #self.last_op = "cross_gnn"
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
                #plan_query= self.sa_atten(plan_query, self.last_plan_query.to(plan_query.device))
                # import pdb;pdb.set_trace()
                
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,#torch.Size([6, 1, 18])
                    plan_reg,#torch.Size([6, 1, 18, 12, 2])
                    plan_status,
                ) = self.layers[i](
                    motion_query, #([6, 900, 6, 256]
                    plan_query, #6, 1, 18, 256]
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)
        #import pdb;pdb.set_trace()
        motion_output = {
            "classification": motion_classification, #[6, 900, 6]
            "prediction": motion_prediction,#[6, 900, 6, 12, 2])
            "period": self.instance_queue.period,#[6, 900]
            "anchor_queue": self.instance_queue.anchor_queue,#torch.Size([6, 900, 11])
        }
        
        # Current perception predicts multi-modal surrounding-agent motion.
        # The latent world model turns those trajectories into twelve future
        # scene tokens and rolls each command-conditioned ego mode forward.
        world_model_output = self.latent_world_model(
            plan_query=plan_query,
            ego_feature=instance_feature[:, num_anchor:],
            agent_features=instance_feature[:, :num_anchor],
            agent_confidence=det_confidence,
            map_features=map_instance_feature_selected,
            map_confidence=map_confidence_selected,
            motion_logits=motion_classification[-1],
            motion_deltas=motion_prediction[-1],
            base_plan_logits=planning_classification[-1],
            base_plan_deltas=planning_prediction[-1],
        )
        enhanced_plan_query = world_model_output["enhanced_plan_query"]

        plan_cls_2th, plan_reg_2th, plan_status_2th = self.refine_2th_layer(
            enhanced_plan_query,
            instance_feature[:, num_anchor:],
            anchor_embed[:, num_anchor:],
        )
        planning_classification_refined.append(plan_cls_2th)
        planning_prediction_refined.append(plan_reg_2th)
        planning_status_refined.append(plan_status_2th)

        # Original MomAD planning output.
        base_classification = planning_classification[-1]
        base_prediction = planning_prediction[-1]
        base_status = planning_status[-1]

        # sigmoid(-4.595) is approximately 0.01.
        final_gate = torch.sigmoid(self.final_fusion_bias)

        if self.world_fusion_enabled:
            # Gradually inject the world-model result into the original MomAD result.
            # gate = 0: original MomAD
            # gate = 1: world-model result
            final_classification = base_classification + final_gate * (
                world_model_output["world_plan_logits"] - base_classification
            )

            world_prediction = world_model_output["world_plan_deltas"]

            if getattr(self, "consistency_aware_fusion", False):
                last_prediction = self.last_final_planning_prediction.to(
                    device=base_prediction.device,
                    dtype=base_prediction.dtype,
                )

                if last_prediction.dim() == 3:
                    last_prediction = last_prediction[:, None, None, :, :]

                # Align consistency gating with evaluation space:
                # planning decoder evaluates cumulative absolute trajectory,
                # not raw per-step deltas.


                base_abs = base_prediction[..., :2].cumsum(dim=-2)
                world_abs = world_prediction[..., :2].cumsum(dim=-2)
                last_abs = last_prediction[..., :2]
                

                base_consist = torch.norm(
                    base_abs - last_abs,
                    dim=-1,
                ).mean(dim=-1, keepdim=True)

                world_consist = torch.norm(
                    world_abs - last_abs,
                    dim=-1,
                ).mean(dim=-1, keepdim=True)

                consistency_scale = torch.where(
                    world_consist <= base_consist,
                    torch.ones_like(world_consist),
                    base_consist / (world_consist + 1e-6),
                ).clamp(min=0.2, max=1.0)

                consistency_scale = consistency_scale[..., None]
            else:
                consistency_scale = 1.0

            if self.time_adaptive_fusion:
                # Near-term planning should stay close to MomAD.
                # Long-horizon steps receive stronger world-model residual correction.
                time_gate = torch.linspace(
                    0.2,
                    1.0,
                    self.ego_fut_ts,
                    device=base_prediction.device,
                    dtype=base_prediction.dtype,
                ).view(1, 1, 1, self.ego_fut_ts, 1)

                final_prediction = base_prediction + final_gate * time_gate * consistency_scale * (
                    world_prediction - base_prediction
                )
            else:
                final_prediction = base_prediction + final_gate * consistency_scale * (
                    world_prediction - base_prediction
                )

            final_status = base_status + final_gate * (
                plan_status_2th - base_status
            )
        else:
            # Strict MomAD semantic-alignment mode.
            final_classification = base_classification
            final_prediction = base_prediction
            final_status = base_status
        
        if getattr(self, "consistency_aware_fusion", False):
            with torch.no_grad():
                final_scores = final_classification.squeeze(1)
                final_mode = final_scores.argmax(dim=-1)
                gather_index = final_mode[:, None, None, None].expand(
                    -1, 1, self.ego_fut_ts, 2
                )
                selected_final_prediction = torch.gather(
                    final_prediction.squeeze(1),
                    dim=1,
                    index=gather_index,
                ).squeeze(1)

                selected_final_prediction = selected_final_prediction.cumsum(dim=-2)

                self.last_final_planning_prediction = selected_final_prediction.detach()

        planning_output = {
            "classification": planning_classification,#[6, 1, 18] #3是command
            "prediction": planning_prediction,#[6, 1, 18, 12, 2]
            "status": planning_status,#[6, 1, 10]
            "classification_refined": planning_classification_refined,#[6, 1, 18] #3是command
            "prediction_refined": planning_prediction_refined,#[6, 1, 18, 12, 2]
            "status_refined": planning_status_refined,#[6, 1, 10]
            "world_classification": world_model_output["world_plan_logits"],
            "world_prediction": world_model_output["world_plan_deltas"],
            "world_scene_flow": world_model_output["scene_flow"],
            "world_future_latents": world_model_output["future_latents"],
            "final_classification": [final_classification],
            "final_prediction": [final_prediction],
            "final_status": [final_status],
            "world_residual_scale": world_model_output["residual_scale"],
            "final_fusion_gate": final_gate,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    

    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        planning_loss_final = self.loss_planning_final(planning_model_outs, data)
        loss.update(planning_loss_final)
        world_model_loss = self.loss_world_model(planning_model_outs, data)
        loss.update(world_model_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output
    
    def loss_planning_final(self, model_outs, data):
        cls_scores = model_outs["final_classification"]
        reg_preds = model_outs["final_prediction"]
        status_preds = model_outs["final_status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_final_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_final_{decoder_idx}": reg_loss,
                    f"planning_loss_status_final_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs",))
    def loss_world_model(self, model_outs, data):
        """Supervise latent rollout with planning and future-scene targets.

        Unlike the previous experimental implementation, the future latents
        are not detached.  Both the auxiliary ego trajectory and the dynamic
        scene-flow loss therefore update the recurrent world model.
        """
        cls = model_outs["world_classification"]
        reg = model_outs["world_prediction"]
        (
            sampled_cls,
            cls_target,
            cls_weight,
            reg_pred,
            reg_target,
            reg_weight,
        ) = self.planning_sampler.sample(
            cls,
            reg,
            data["gt_ego_fut_trajs"],
            data["gt_ego_fut_masks"],
            data,
        )

        sampled_cls = sampled_cls.flatten(end_dim=1)
        cls_target = cls_target.flatten(end_dim=1)
        cls_weight = cls_weight.flatten(end_dim=1)
        cls_loss = self.plan_loss_cls(
            sampled_cls, cls_target, weight=cls_weight
        ) * self.world_plan_cls_weight

        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.flatten(end_dim=1)
        # Absolute-position supervision prevents small per-step bias from
        # accumulating into a large 4--6 second planning error.
        reg_pred = reg_pred.cumsum(dim=-2)
        reg_target = reg_target.cumsum(dim=-2)
        horizon_weight = torch.linspace(
            1.0,
            self.long_horizon_weight,
            self.ego_fut_ts,
            device=reg_pred.device,
            dtype=reg_pred.dtype,
        )
        weighted_mask = reg_weight * horizon_weight.unsqueeze(0)
        reg_loss = self.plan_loss_reg(
            reg_pred,
            reg_target,
            weight=weighted_mask.unsqueeze(-1),
        ) * self.world_plan_reg_weight

        scene_flow_pred = model_outs["world_scene_flow"]
        flow_target = scene_flow_pred.new_zeros(scene_flow_pred.shape)
        flow_mask = scene_flow_pred.new_zeros(scene_flow_pred.shape[:2])
        gt_agent_trajs = data["gt_agent_fut_trajs"]
        gt_agent_masks = data["gt_agent_fut_masks"]
        for batch_idx in range(scene_flow_pred.shape[0]):
            agent_traj = gt_agent_trajs[batch_idx].to(scene_flow_pred.device)
            agent_mask = gt_agent_masks[batch_idx].to(scene_flow_pred.device)
            if agent_traj.numel() == 0:
                continue
            agent_traj = agent_traj[:, : self.ego_fut_ts]
            agent_mask = agent_mask[:, : self.ego_fut_ts]
            valid_count = agent_mask.sum(dim=0)
            flow_target[batch_idx] = (
                agent_traj * agent_mask.unsqueeze(-1)
            ).sum(dim=0) / valid_count.clamp(min=1).unsqueeze(-1)
            flow_mask[batch_idx] = valid_count.gt(0).to(flow_mask.dtype)
        flow_error = F.smooth_l1_loss(
            scene_flow_pred, flow_target, reduction="none"
        ).sum(dim=-1)
        scene_flow_loss = (
            (flow_error * flow_mask).sum() / flow_mask.sum().clamp(min=1)
        ) * self.world_scene_flow_weight

        future_latents = model_outs["world_future_latents"]
        required_world_keys = (
            "gt_world_model_agent_states",
            "gt_world_model_agent_masks",
            "gt_world_model_ego_states",
            "gt_world_model_ego_masks",
        )
        if all(key in data for key in required_world_keys):
            target = self.latent_world_model.encode_future_world_targets(
                data["gt_world_model_agent_states"],
                data["gt_world_model_agent_masks"],
                data["gt_world_model_ego_states"],
                data["gt_world_model_ego_masks"],
                future_latents.device,
            )

            batch_size = reg.shape[0]
            world_by_command = reg.squeeze(1).reshape(
                batch_size,
                3,
                self.ego_fut_mode,
                self.ego_fut_ts,
                2,
            )
            batch_indices = torch.arange(batch_size, device=reg.device)
            command = data["gt_ego_fut_cmd"].argmax(dim=-1)
            command_reg = world_by_command[batch_indices, command]
            command_absolute = command_reg.cumsum(dim=-2)
            gt_absolute = data["gt_ego_fut_trajs"].cumsum(dim=-2)
            gt_mask = data["gt_ego_fut_masks"]
            distance = torch.linalg.norm(
                command_absolute - gt_absolute[:, None], dim=-1
            )
            distance = (distance * gt_mask[:, None]).sum(dim=-1) / gt_mask.sum(
                dim=-1, keepdim=True
            ).clamp(min=1)
            best_mode = distance.argmin(dim=-1)
            global_mode = command * self.ego_fut_mode + best_mode
            selected_latent = future_latents[batch_indices, global_mode]

            steps = min(selected_latent.shape[1], target["latent"].shape[1])
            selected_latent = selected_latent[:, :steps]
            target_latent = target["latent"][:, :steps]
            latent_mask = target["valid_mask"][:, :steps]
            latent_error = F.smooth_l1_loss(
                selected_latent,
                target_latent.detach(),
                reduction="none",
            ).mean(dim=-1)
            latent_loss = (
                (latent_error * latent_mask).sum()
                / latent_mask.sum().clamp(min=1)
            ) * self.world_latent_weight

            pred_ego_state, pred_agent_state = (
                self.latent_world_model.decode_future_world_states(selected_latent)
            )
            teacher_ego_state, teacher_agent_state = (
                self.latent_world_model.decode_future_world_states(target_latent)
            )

            def masked_state_loss(prediction, state_target, state_mask):
                error = F.smooth_l1_loss(
                    prediction, state_target, reduction="none"
                ).mean(dim=-1)
                return (error * state_mask).sum() / state_mask.sum().clamp(min=1)

            ego_target = target["ego_state"][:, :steps]
            agent_target = target["agent_state"][:, :steps]
            agent_mask = target["agent_valid_mask"][:, :steps]
            state_reconstruction_loss = (
                masked_state_loss(pred_ego_state, ego_target, latent_mask)
                + masked_state_loss(pred_agent_state, agent_target, agent_mask)
                + masked_state_loss(teacher_ego_state, ego_target, latent_mask)
                + masked_state_loss(teacher_agent_state, agent_target, agent_mask)
            ) * (self.world_state_reconstruction_weight / 4.0)
        else:
            # Allows a smoke test with legacy infos, but a real training log
            # must show non-zero latent/state losses from the new converter.
            latent_loss = future_latents.sum() * 0.0
            state_reconstruction_loss = future_latents.sum() * 0.0

        # A small diversity term discourages six modes under one command from
        # collapsing onto the same 6-second endpoint.
        world_absolute = reg.squeeze(1).reshape(
            reg.shape[0],
            3,
            self.ego_fut_mode,
            self.ego_fut_ts,
            2,
        ).cumsum(dim=-2)
        endpoints = world_absolute[..., -1, :]
        pair_distance = torch.cdist(endpoints, endpoints)
        off_diagonal = 1.0 - torch.eye(
            self.ego_fut_mode,
            device=pair_distance.device,
            dtype=pair_distance.dtype,
        )
        diversity_loss = (
            torch.exp(-pair_distance / 2.0) * off_diagonal
        ).sum() / off_diagonal.sum().clamp(min=1) / endpoints.shape[0] / 3
        diversity_loss = diversity_loss * self.world_diversity_weight

        return {
            "world_model_loss_cls": cls_loss,
            "world_model_loss_reg": reg_loss,
            "world_model_loss_scene_flow": scene_flow_loss,
            "world_model_loss_latent": latent_loss,
            "world_model_loss_state_reconstruction": state_reconstruction_loss,
            "world_model_loss_diversity": diversity_loss,
        }

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        # Decode exactly the same fused branch that receives the final loss.
        #planning_decode_output = dict(planning_output)
        #planning_decode_output["classification"] = planning_output[
          #  "final_classification"
        #]
       # planning_decode_output["prediction"] = planning_output[
          #  "final_prediction"
        #]
        #planning_decode_output["status"] = planning_output["final_status"]
        
                # Select which planning branch is decoded during evaluation.
        # base: original MomAD first-stage planning output
        # final: current world-model fused output
        # refined: second-stage refined planning output
        # world: pure latent-world-model planning output
        planning_decode_output = dict(planning_output)
        eval_branch = getattr(self, "eval_planning_branch", "final")

        if eval_branch == "base":
            planning_decode_output["classification"] = planning_output["classification"]
            planning_decode_output["prediction"] = planning_output["prediction"]
            planning_decode_output["status"] = planning_output["status"]

        elif eval_branch == "final":
            planning_decode_output["classification"] = planning_output["final_classification"]
            planning_decode_output["prediction"] = planning_output["final_prediction"]
            planning_decode_output["status"] = planning_output["final_status"]

        elif eval_branch == "refined":
            planning_decode_output["classification"] = planning_output["classification_refined"]
            planning_decode_output["prediction"] = planning_output["prediction_refined"]
            planning_decode_output["status"] = planning_output["status_refined"]

        elif eval_branch == "world":
            planning_decode_output["classification"] = [planning_output["world_classification"]]
            planning_decode_output["prediction"] = [planning_output["world_prediction"]]
            planning_decode_output["status"] = planning_output["status"]

        else:
            raise ValueError(f"Unsupported eval_planning_branch: {eval_branch}")

        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_decode_output,
            data,
        )

        return motion_result, planning_result
