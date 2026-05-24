# LVR Stage-1 AMLT job: 1 node × 8 × A100 40GB

Target requested for Singularity:

```text
name: msrresrchvc
workspace: gcr-singularity-resrch
GPU: A100 40GB
shape: 1 node × 8 GPUs
```

The job YAML is in the GitHub repo:

```text
amlt/lvr_stage1_1node8a10040_singularity.yaml
```

Submit from an AMLT project directory:

```bash
amlt run /path/to/lvr/amlt/lvr_stage1_1node8a10040_singularity.yaml
```

## Expected mounted data layout

The YAML assumes storage is mounted at `/mnt/lvr`, and data is under:

```text
/mnt/lvr/dataset/Visual-CoT/
  annotations/meta_data_lvr_sft_stage1.local.json
  annotations/viscot_363k_lvr_formatted.json
  annotations/viscot_sroie_dude_lvr_formatted.json
  images/cot_image_data/...

/mnt/lvr/dataset/ViRL39K/
  annotations/virl39k.json
  images/...
```

If your storage/container differs, edit these YAML defaults only:

```yaml
DATA_ROOT: /mnt/lvr/dataset
OUTPUT_ROOT: /mnt/lvr/lvr_runs
CACHE_ROOT: /mnt/lvr/lvr_cache
storage.lvr.container_name: amulet
storage.lvr.mount_dir: /mnt/lvr
```

## Training command used by the job

The actual training command is the same shape as the local command, adapted to 8 GPUs and the GitHub-cloned repo:

```bash
cd ${REPO_DIR}
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
export WANDB_MODE=offline
export CACHE_DIR=${CACHE_ROOT}/hf

deepspeed --num_gpus=${GPUS} src/train/train_lvr.py \
  --run_name "Stage1_8xA10040_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128" \
  --coconut True \
  --loss_lvr_fct mse \
  --deepspeed scripts/zero3_offload.json \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --data_path ${DATA_ROOT}/Visual-CoT/annotations/meta_data_lvr_sft_stage1.local.json \
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
  --output_dir ${OUTPUT_ROOT}/stage1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --image_min_pixels $((128 * 28 * 28)) \
  --image_max_pixels $((5120 * 28 * 28)) \
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
  --max_packed_tokens $((4 * 4096)) \
  --random_seed 42 \
  --long_seq_threshold 4096 \
  --max_instance_per_batch 4
```

Notes:
- `--disable_flash_attn2 True` is intentional; this repo/env is validated without `flash-attn`.
- `gradient_accumulation_steps=8` is for 8 GPUs. The earlier single-A100 command used 64 to approximate the same global batch.
- A100 40GB may still be tight with `image_max_pixels=5120*28*28`; if it OOMs, first try `image_max_pixels=$((2560 * 28 * 28))`.
