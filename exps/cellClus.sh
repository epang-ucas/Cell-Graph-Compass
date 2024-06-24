#!/bin/bash

source ~/.bashrc
conda activate CGC

[ -z "${batchsize}" ] && batchsize=32
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seq_max_len}" ] && seq_max_len=1201

[ -z "${lr}" ] && lr=1e-4
[ -z "${warmup_step}" ] && warmup_step=500
[ -z "${decay_step}" ] && decay_step=300
[ -z "${decay_ratio}" ] && decay_ratio=0.95
[ -z "${save_interval}" ] && save_interval=1000
[ -z "${validate_interval}" ] && validate_interval=1000
[ -z "${total_epoch}" ] && total_epoch=30

[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${num_worker}" ] && num_worker=12
[ -z "${model_name}" ] && model_name=gene_model
[ -z "${seed}" ] && seed=42
[ -z "${total_step}" ] && total_step=8000000

[ -z "${task}" ] && task=single_cell_task
[ -z "${loss}" ] && loss=single_cell_loss
[ -z "${arch}" ] && arch=single_cell_model

[ -z "${user_dir}" ] && user_dir=../src
[ -z "${unicore_path}" ] && unicore_path=~/miniconda3/envs/CGC/bin/unicore-train
[ -z "${data_path}" ] && data_path=../scData/example_datasets/cellclus_PCortex
[ -z "${vocab_path}" ] && vocab_path=../scData/bioFeature_embs/vocab.json
[ -z "${text_emb_path}" ] && text_emb_path=../scData/bioFeature_embs/embeding.pickle
[ -z "${pretrain}" ] && pretrain=../scData/pretrain_weights/CGCompass_pretrain_weights.pt

echo "lr" $lr
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "decay_step" $decay_step

echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "mask_ratio" $mask_ratio

name=test
savedir=savedir/cellclus/covid/$name
mkdir -p $savedir/logs

echo "savedir" $savedir

# train

tmp_dir=`mktemp -d`

task_params=""
echo "start training"
echo source ~/.bashrc 
echo "module load cuda/11.7.1-cudnn-v8.5.0"
export PYTHONPATH=$user_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python ${unicore_path} $data_path  $vocab_path --user-dir $user_dir \
       --num-workers $num_worker --ddp-backend=no_c10d \
       --task $task --loss $loss --arch $arch \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 1.0 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay \
       --batch-size $batchsize --update-freq $update_freq --seed $seed --tensorboard-logdir ${savedir}/tsb/ \
       --max-update $total_step --max-epoch $total_epoch --log-interval 1 --log-format simple \
       --save-interval-updates $save_interval --validate-interval-updates $validate_interval --keep-interval-updates 40 --no-epoch-checkpoints  \
       --save-dir ${savedir} --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --data-buffer-size 512 \
       --do_mvg --do_mvc --do_ecs --use_batch_labels --do_dab \
       --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 \
       --seq_max_len 1201 --mask_ratio 0.4 \
       --use_fast_transformer --bf16-sr --bf16 \
       --shuffle --ecs_threshold 0.8 --embed --use_gnn --use_graph --text_emb_path=$text_emb_path --pretrain $pretrain

rm -rf $tmp_dir
echo ""
echo "rm -rf $tmp_dir"

# test

# model_path=../scData/downstream_weights/cellClus/PCortex/model_CGC.pt

# export PYTHONPATH=$user_dir:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES="2"
# python ../src/test_clus.py $data_path $model_path $savedir $vocab_path --batch_size $batchsize \
#     --use_batch_labels --do_ecs --do_mvc --do_mvg --do_dab \
#     --seq_max_len 1201  --mask_ratio 0.4 --ecs_threshold 0.8 \
#     --use_fast_transformer --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 \
#     --embed --text_emb_path=$text_emb_path --use_gnn --use_graph