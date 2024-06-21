#!/bin/bash

source ~/.bashrc
module load cuda/11.7.1-cudnn-v8.5.0
conda activate /work1/superbrain/chenfang2/envs/CGC

[ -z "${batchsize}" ] && batchsize=4
[ -z "${update_freq}" ] && update_freq=1
# [ -z "${batchsize}" ] && batchsize=8
# [ -z "${update_freq}" ] && update_freq=2
[ -z "${seq_max_len}" ] && seq_max_len=3001

[ -z "${lr}" ] && lr=1e-4
[ -z "${warmup_step}" ] && warmup_step=500 #500
[ -z "${decay_step}" ] && decay_step=100
[ -z "${decay_ratio}" ] && decay_ratio=0.995 #995
[ -z "${save_interval}" ] && save_interval=500
[ -z "${validate_interval}" ] && validate_interval=500
[ -z "${total_epoch}" ] && total_epoch=30 #30

[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${num_worker}" ] && num_worker=12
[ -z "${model_name}" ] && model_name=gene_model
[ -z "${seed}" ] && seed=42
[ -z "${total_step}" ] && total_step=8000000

[ -z "${task}" ] && task=single_cell_task
[ -z "${loss}" ] && loss=single_cell_loss
[ -z "${arch}" ] && arch=single_cell_model

[ -z "${user_dir}" ] && user_dir=/work1/superbrain/chenfang2/CGC/src
[ -z "${unicore_path}" ] && unicore_path=/home4/superbrain/chenfang2/anaconda3/bin/unicore-train

[ -z "${data_path}" ] && data_path=/work1/superbrain/chenfang2/lmdb/cellAnno_mye
[ -z "${vocab_path}" ] && vocab_path=/train14/superbrain/zlhu12/lmdb_human_sc_total_new
[ -z "${text_emb_path}" ] && text_emb_path=/train14/superbrain/chenfang2/train14_zlhu/gene_model_new/src/data/embeding.pickle
# [ -z "${pretrain}" ] && pretrain=/train14/superbrain/zlhu12/gene_model_1_gf/exps/savedir/new_train/pretrain_human_all_target/checkpoint_best.pt
# [ -z "${pretrain}" ] && pretrain=/train14/superbrain/zlhu12/gene_model_1_gf/exps/savedir/new_train/pretrain_human_all_target/checkpoint_3_160000.pt

[ -z "${pretrain}" ] && pretrain=/train14/superbrain/zlhu12/gene_model_new/exps/savedir/pretrain/human_all_gnn_e/checkpoint_30_140000.pt
# [ -z "${pretrain}" ] && pretrain=/train14/superbrain/zlhu12/gene_model_new/exps/savedir/pretrain/human_all_gnn_e/checkpoint_39_180000.pt
# [ -z "${pretrain}" ] && pretrain=/train14/superbrain/zlhu12/gene_model_new/exps/savedir/pretrain/human_all_gnn_e/checkpoint_50_230000.pt


echo "lr" $lr
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "decay_step" $decay_step

echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "mask_ratio" $mask_ratio

name=test
savedir=savedir/cellAnno/mye/$name
mkdir -p $savedir

echo "savedir" $savedir

# train

# tmp_dir=`mktemp -d`

# task_params=""
# echo "start training"
# echo source ~/.bashrc 
# echo "module load cuda/11.7.1-cudnn-v8.5.0"
# export PYTHONPATH=$user_dir:$PYTHONPATH
# python ${unicore_path} $data_path  $vocab_path --user-dir $user_dir \
#        --num-workers $num_worker --ddp-backend=no_c10d \
#        --task $task --loss $loss --arch $arch \
#        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 1.0 --allreduce-fp32-grad  \
#        --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay \
#        --batch-size $batchsize --update-freq $update_freq --seed $seed --tensorboard-logdir ${savedir}/tsb/ \
#        --max-update $total_step --max-epoch $total_epoch --log-interval 1 --log-format simple \
#        --save-interval-updates $save_interval --validate-interval-updates $validate_interval --keep-interval-updates 40 --no-epoch-checkpoints  \
#        --save-dir ${savedir} --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --data-buffer-size 128 \
#        --do_cls \
#        --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 --use_gnn --use_graph --mask_ratio 0.4 \
#        --seq_max_len 3001 \
#        --use_fast_transformer --bf16-sr --bf16 \
#        --shuffle --embed --text_emb_path $text_emb_path
# rm -rf $tmp_dir
# echo ""
# echo "rm -rf $tmp_dir"


# test

[ -z "${model_path}" ] && model_path=/work1/superbrain/chenfang2/downstream_results/cellAnno/mye/CGC.pt
savedir=savedir=/work1/superbrain/chenfang2/tmp

[ -z "${batch_size}" ] && batch_size=8
export PYTHONPATH=$user_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"
python ../src/test_anno.py $data_path $model_path $savedir $vocab_path --batch_size $batch_size \
    --use_fast_transformer --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 \
    --do_cls --seq_max_len 3001 --mask_ratio 0 \
    --embed --text_emb_path $text_emb_path --use_gnn --use_graph 