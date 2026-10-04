export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

DATA_ROOT=${DATA_ROOT:-./data/nuscenes}

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path /data/share/nuscenes \
#     --canbus /data/share/nuscenes \
#     --out-dir ./data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0-mini

python tools/data_converter/nuscenes_converter_MomAD_World_model_6s.py nuscenes \
    --root-path "$DATA_ROOT" \
    --canbus "$DATA_ROOT" \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0

#python tools/data_converter/nuscenes_converter_6s.py nuscenes --root-path /data/share/nuscenes --canbus /data/share/nuscenes --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0
