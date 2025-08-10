export WANDB_API_KEY='0fc0cd9916b988b325b1c64da2f5bbaf48f8e80b'
export HYDRA_FULL_ERROR=1
source az_info_setup.sh

ray start --head --num-gpus 8

bash train_grpo_math_tune_ray.sh \
    --model_name Qwen2.5-7B --max_response_length 8192  \
    --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0 \
    --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 \
    --rollout_tp 2 --save_freq 10  
