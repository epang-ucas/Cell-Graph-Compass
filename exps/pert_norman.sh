# !/bin/bash

source ~/.bashrc
conda activate CGC

[ -z "${batchsize}" ] && batchsize=16
[ -z "${update_freq}" ] && update_freq=2

[ -z "${lr}" ] && lr=1e-4 #-6
[ -z "${warmup_step}" ] && warmup_step=1000
[ -z "${decay_step}" ] && decay_step=765
[ -z "${decay_ratio}" ] && decay_ratio=0.9
[ -z "${save_interval}" ] && save_interval=1500
[ -z "${validate_interval}" ] && validate_interval=900
[ -z "${total_epoch}" ] && total_epoch=30

[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${num_worker}" ] && num_worker=12
[ -z "${model_name}" ] && model_name=gene_perturbation
[ -z "${seed}" ] && seed=42
[ -z "${total_step}" ] && total_step=8000000

#[ -z "${sd_prob}" ] && sd_prob=0.75
[ -z "${task}" ] && task=gene_pert
[ -z "${loss}" ] && loss=pert_loss
[ -z "${arch}" ] && arch=pert_model

[ -z "${user_dir}" ] && user_dir=../src
[ -z "${unicore_path}" ] && unicore_path=~/miniconda3/envs/CGC/bin/unicore-train
[ -z "${data_path}" ] && data_path=../scData/example_datasets/Norman
[ -z "${vocab_path}" ] && vocab_path=../scData/bioFeature_embs/vocab.json
[ -z "${text_emb_path}" ] && text_emb_path=../scData/bioFeature_embs/embeding.pickle
[ -z "${edge_path}" ] && edge_path=../scData/example_datasets/Norman/edge.pickle
[ -z "${pretrain}" ] && pretrain=../scData/pretrain_weights/CGCompass_pretrain_weights.pt

echo "lr" $lr
echo "batchsize" $batchsize
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "decay_step" $decay_step

echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "mask_ratio" $mask_ratio

name=test
savedir=savedir/pert/$name
mkdir -p $savedir/logs

echo "savedir" $savedir

# train

tmp_dir=`mktemp -d`

task_params=""
echo "start training"
echo source ~/.bashrc 
echo "module load cuda/11.7.1-cudnn-v8.5.0"
export PYTHONPATH=$user_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"
python ${unicore_path} $data_path $vocab_path --user-dir $user_dir \
       --num-workers $num_worker --ddp-backend=no_c10d \
       --task $task --loss $loss --arch $arch \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 1.0 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay \
       --batch-size $batchsize --update-freq $update_freq --seed $seed --tensorboard-logdir ${savedir}/tsb/ \
       --max-update $total_step --max-epoch $total_epoch --log-interval 1 --log-format simple \
       --save-interval-updates $save_interval --validate-interval-updates $validate_interval --keep-interval-updates 40 --no-epoch-checkpoints  \
       --save-dir ${savedir} --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --data-buffer-size 128 \
       --do_mvg --use_batchnorm --text_emb_path $text_emb_path --edge_path $edge_path \
       --use_fast_transformer --shuffle --bf16-sr --bf16  --dropout 0.2 \
       --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 
   

echo ""
echo "rm -rf $tmp_dir"

## test

# model_path=../scData/downstream_weights/scPert/model_CGC.pt

# export PYTHONPATH=$user_dir:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES="0"
# python ../src/test_pert.py $data_path $vocab_path $model_path $savedir \
#     --batch_size $batchsize --pool_size 300 \
#     --do_mvg --use_batchnorm --dropout 0.0 \
#     --use_fast_transformer --nlayers 12 --nhead 8 --d_model 512 --d_hid 512 \
#     --use_gnn --use_graph --embed --text_emb_path $text_emb_path --edge_path $edge_path