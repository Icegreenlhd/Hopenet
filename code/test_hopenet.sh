set -ex

python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_0.0_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl &&
python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_0.01_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl &&
python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_0.1_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl &&
python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_1.0_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl &&
python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_2.0_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl &&
python test_hopenet.py --gpu 0 --snapshot output/snapshots/model_hopenet_lr_1e-05_alpha_4.0_epoch-num_5_gpu0_batch-size_32_300W_LP/epoch_5.pkl