# Fine-tune the independent 6s latent-world-model experiment from the
# MomAD/roboAD 6s checkpoint configured in the Python config.
bash ./tools/dist_train.sh \
   projects/configs/MomAD_small_stage2_MomAD_World_model_6s_v2.py \
   1 \
   --deterministic

