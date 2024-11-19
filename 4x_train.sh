# python main_train.py --trainer_config configs/trainer/4x.yaml --data_config configs/scenes/bunker/bunker_4x_H2_train.yaml \
#     --resume true --max_epochs 200 --batch_size 8

python main_train.py --trainer_config configs/trainer/4x_render.yaml --data_config configs/scenes/bunker/bunker_4x_H2_train.yaml \
    --max_epochs 200 --batch_size 8
