"""Opt-in architecture example; prepare model-only warm-start weights first.

Inherits the public prototype's optimizer/schedule, NOT any private training
continuation. Use tools/prepare_current_ego_config.py for path-checked config
generation; it also relocates every nested work_dir in the selected baseline.
"""

_base_ = './MomAD_small_stage2_MomAD_World_model_6s_v2_oracle_mode_reg02_resume_repro.py'

work_dir = 'work_dirs/momworld_current_ego_6s'
resume_from = None
load_from = 'work_dirs/momworld_current_ego_6s/init.pth'

model = dict(head=dict(motion_plan_head=dict(use_current_ego_status=True)))
data = dict(
    train=dict(work_dir=work_dir),
    val=dict(work_dir=work_dir),
    test=dict(work_dir=work_dir),
)
custom_hooks = [
    dict(
        type='FreezeForLatentWorldModelMomAD6sHook',
        train_keywords=(
            'motion_plan_head.latent_world_model',
            'motion_plan_head.final_fusion_bias',
            'motion_plan_head.ego_state_encoder',
        ),
        freeze_batch_norm=True,
        priority='VERY_HIGH',
    ),
]
