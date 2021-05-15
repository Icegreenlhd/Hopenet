set -ex

python train_alexnet.py --lr 0.00001 --alpha 4    --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP  &&
python train_alexnet.py --lr 0.00001 --alpha 2    --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP  &&
python train_alexnet.py --lr 0.00001 --alpha 1    --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP  &&
python train_alexnet.py --lr 0.00001 --alpha 0.1  --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP  &&
python train_alexnet.py --lr 0.00001 --alpha 0.01 --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP  &&
python train_alexnet.py --lr 0.00001 --alpha 0    --gpu 1 --num_epochs 5 --batch_size 32 --output_string 300W_LP