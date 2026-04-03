# Training

This guide covers the default training workflow used by `scripts/train/train.sh`.
Run all commands below from the repository root after completing environment
setup in `README.md`.

## Data Preparation

The default training mix reads these paths:

- `data/train/spar_234k.json`
- `data/train/llava_hound_64k.json`
- `data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json`
- `data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json`

Download annotations and map them to the paths above:

```bash
hf download Journey9ni/SpatialStackData \
  --repo-type dataset \
  --include "annotations/**" \
  --local-dir data

mkdir -p ./data/train
mkdir -p ./data/vlm3r/annotations/vsibench_train
mkdir -p ./data/vsi_590k/annotations

ln -sfn ../annotations/spar_234k.json ./data/train/spar_234k.json
ln -sfn ../annotations/llava_hound_64k.json ./data/train/llava_hound_64k.json
ln -sfn ../../../annotations/merged_qa_scannet_train.json \
  ./data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json
ln -sfn ../../annotations/vsi_appearance_order_vsibench_scannet.json \
  ./data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json
```

Download the media used by the same default mix.

SPAR:

```bash
mkdir -p ./data/media/spar

hf download jasonzhango/SPAR-7M \
  --repo-type dataset \
  --revision 976c19177468eabe64e9e2dd0f0450cd32dacc1f \
  --include "*.tar.gz" \
  --local-dir ./data/media/spar

( cd ./data/media/spar && for f in *.tar.gz; do tar -I pigz -xf "$f"; done )
```

LLaVA-Hound:

```bash
mkdir -p ./data/media/llava_hound

hf download ShareGPTVideo/train_video_and_instruction \
  --repo-type dataset \
  --include "train_300k/**" \
  --local-dir ./data/media/llava_hound

mkdir -p ./data/media/llava_hound/frames
find ./data/media/llava_hound/train_300k -maxdepth 1 -name 'chunk_*.tar.gz' -print0 \
| xargs -0 -n1 -P"$(nproc)" -I{} tar -I pigz -x -f "{}" -C ./data/media/llava_hound/frames
```

VLM-3R ScanNet video:

```bash
mkdir -p ./data/vlm3r/media/scannet

hf download Journey9ni/aweb \
  --repo-type dataset \
  --include "ScanNet/videos/train/**" \
  --local-dir ./data/vlm3r/media/scannet

mv ./data/vlm3r/media/scannet/ScanNet/videos/train \
  ./data/vlm3r/media/scannet/videos
```

VSI-590K reuses the same ScanNet videos:

```bash
mkdir -p ./data/vsi_590k/media
ln -sfn ../../vlm3r/media/scannet/videos ./data/vsi_590k/media/scannet
```

## Launch Training

Use the unified training script:

```bash
bash scripts/train/train.sh
```

Before running, edit these parameters in `scripts/train/train.sh`:

- `MODEL_PATH`: base VLM path or HF id (default: `Qwen/Qwen2.5-VL-3B-Instruct`)
- `GEOMETRY_ENCODER_PATH`: geometry encoder path or HF id (default: `facebook/VGGT-1B`)
- `OUTPUT_DIR`: checkpoint/log directory (default: `./output/spatialstack_train`)
- `CACHE_DIR`: model cache directory (default: `./cache`)
- `DATASETS`: training datasets and sampling ratio string  
  default: `spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50`
- `LR`: learning rate (default: `1e-5`)
- `total_batch_size`: global batch size used to compute `gradient_accumulation_steps`
- `GEOMETRY_ENCODER_TYPE`: geometry encoder type (default: `vggt`)
- `feature_fusion_method`: fusion type (default: `deepstack_language_add`)
- `geometry_fusion_layers`: decoder layers for geometry fusion (default: `0 1 2`)
- `geometry_encoder_layers`: geometry feature layers to extract (default: `11 17 23`)
- `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `CUDA_VISIBLE_DEVICES`: distributed launch controls (optional)
