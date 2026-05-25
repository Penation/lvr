# LVR Stage-1 AMLT job: 1 node × 8 × MI200 64GB

This file documents the MI200 ROCm job for reproducing LVR Stage-1 SFT on Singularity.

## Target

```text
service: sing
workspace: wsgcrrbt
target: msrresrchvc
resource group: gcr-singularity-resrch
GPU: MI200 64GB
shape: 1 node × 8 GPUs
AMLT SKU: 1x64G8-MI200-xGMI
SLA: Premium
```

Verified locally with:

```bash
amlt cache expand-sku -t msrresrchvc 64G8-MI200-xGMI --sla Premium
# ND48as_MI200_v4  64G8-MI200-xGMI
```

## Data and outputs

Blob storage is mounted as `/mnt/lvr` from storage account `azsussc`, container `v-bochengpan`.

Training data:

```text
/mnt/lvr/2026vlm/datasets/Visual-CoT/
/mnt/lvr/2026vlm/datasets/ViRL39K/
```

Stage-1 output/checkpoints:

```text
/mnt/lvr/2026vlm/ckpts/lvr/stage1-mi200
azsussc/v-bochengpan/2026vlm/ckpts/lvr/stage1-mi200
```

A per-job stdout copy is written to:

```text
/mnt/lvr/2026vlm/ckpts/lvr/lvr-stage1-1node8gpu-mi20064-no-flash-v1.log
```

## Submit

```bash
cd /home/v-bochengpan/data/amlt/lvr
amlt run lvr_stage1_1node8mi20064_singularity.yaml --sla Premium
```

## Training command shape

The actual command runs from a GitHub clone of `https://github.com/Penation/lvr.git` on `main` and keeps the Stage-1 settings aligned with the paper/code:

```bash
python -m deepspeed --num_gpus=8 src/train/train_lvr.py \
  --run_name "Stage1_8xMI20064_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128" \
  --coconut True \
  --loss_lvr_fct mse \
  --deepspeed scripts/zero3_offload.json \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --data_path /mnt/lvr/2026vlm/cache/lvr-mi200/run_data/meta_data_lvr_sft_stage1.mi200.json \
  --remove_unused_columns False \
  --lvr_head False \
  --freeze_vision_tower True \
  --freeze_merger True \
  --freeze_llm False \
  --max_steps 2500 \
  --learning_rate 1e-5 \
  --loss_lvr_lambda 0.1 \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --online_checkpoint False \
  --output_dir /mnt/lvr/2026vlm/ckpts/lvr/stage1-mi200 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --image_min_pixels 100352 \
  --image_max_pixels 4014080 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 False \
  --gradient_checkpointing True \
  --report_to none \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 10 \
  --dataloader_num_workers 8 \
  --enable_data_packing True \
  --max_packed_tokens 16384 \
  --random_seed 42 \
  --long_seq_threshold 4096 \
  --max_instance_per_batch 4
```

## Notes

- This job intentionally uses `--disable_flash_attn2 True` for the first MI200 attempt. Hugging Face AMD FlashAttention2 guidance commonly lists MI210/MI250/MI300 support; MI200 availability is not guaranteed, so SDPA is the conservative path.
- The base image is `rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.6.0` because this AMLT install does not list an `amlt-sing/acpt-rocm...` base image via `amlt cache base-images`.
- The existing A100 job writes to `stage1`; this MI200 job writes to `stage1-mi200` so the two jobs do not collide.
