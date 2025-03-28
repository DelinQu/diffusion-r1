# DEBUG=true
if [ "$DEBUG" = true ]; then
  GPUS=1
  wandb_enable=false
  ACCELERATE_ARGS="--num_machines 1 --num_processes 1 --mixed_precision=bf16 --dynamo_backend=no"
fi

# distributed settings
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
wandb_enable=${wandb_enable:-true}

# set environments
# export TRITON_CACHE_DIR=/cpfs01/shared/optimal/vla_ptm/.triton
source scripts/env.sh

ACCELERATE_ARGS=${ACCELERATE_ARGS:-"--main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --num_machines ${NODES} --num_processes=${GPUS} --multi_gpu \
  --mixed_precision=no --dynamo_backend=no"}

dataset=dataset
lr=5e-5
epoch=200
run_name=grpo_${dataset}_gpu${GPUS}_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_ep${epoch}

python grpo/train_grpo.py \
    --num_epochs=$epoch \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=$(($PER_DEVICE_BATCH_SIZE * 2)) \
    --train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=False \
    --per_prompt_stat_tracking_buffer_size=64 \
    --tracker_project_name="diffusion-r1" \
    --log_with=wandb \
    --save_freq 5 \
    --num_checkpoint_limit 2 \
    --train_learning_rate $lr \
    --logdir outputs \
    --run_name $run_name