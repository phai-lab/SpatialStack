#!/usr/bin/env python3

import argparse
import base64
import copy
import json
from io import BytesIO
from pathlib import Path
from typing import List

import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

try:
    from qwen_vl_utils import extract_vision_info
except ImportError as exc:
    raise RuntimeError("qwen_vl_utils is required. Please install dependencies first.") from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = "Journey9ni/SpatialStack-Qwen2.5-4B"


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample inference for SpatialStack models.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"HF model id or local checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--prompt", required=True, help="User prompt text")

    visual_group = parser.add_mutually_exclusive_group(required=True)
    visual_group.add_argument("--image", type=str, help="Path to one image")
    visual_group.add_argument("--image-dir", type=str, help="Directory of images (sorted by filename)")
    visual_group.add_argument("--video", type=str, help="Path to one video")

    parser.add_argument("--device", default="cuda:0", help="Runtime device, e.g. cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-num-frames", type=int, default=32, help="Max frames for video/image-dir")
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--add-frame-index", action="store_true", help="Insert 'Frame-i:' tokens before each image")
    parser.add_argument("--no-flash-attn2", action="store_true", help="Disable flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON path")
    return parser.parse_args()


def sample_indices(total: int, max_count: int) -> np.ndarray:
    if total <= max_count:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, max_count, dtype=int)


def load_visuals(args) -> List[Image.Image]:
    if args.image:
        return [Image.open(args.image).convert("RGB")]

    if args.image_dir:
        image_dir = Path(args.image_dir)
        files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        if not files:
            raise ValueError(f"No image files found in {args.image_dir}")
        indices = sample_indices(len(files), args.max_num_frames)
        return [Image.open(files[i]).convert("RGB") for i in indices]

    vr = decord.VideoReader(args.video)
    indices = sample_indices(len(vr), args.max_num_frames)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in indices]


def image_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(prompt: str, visuals: List[Image.Image], add_frame_index: bool):
    content = []
    for idx, img in enumerate(visuals):
        if add_frame_index:
            content.append({"type": "text", "text": f"Frame-{idx}: "})
        content.append({"type": "image", "image": image_to_data_uri(img)})
    content.append({"type": "text", "text": prompt})

    return [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]]


def prepare_visual_inputs(messages, processor):
    image_inputs = []
    geometry_encoder_inputs = []

    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size

    for message in messages:
        vision_info = extract_vision_info(message)
        cur_geo_inputs = []

        for ele in vision_info:
            if "image" not in ele:
                continue
            image = ele["image"]
            if isinstance(image, str) and "base64," in image:
                _, base64_data = image.split("base64,", 1)
                with BytesIO(base64.b64decode(base64_data)) as bio:
                    image = copy.deepcopy(Image.open(bio))
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type: {type(image)}")

            image = load_and_preprocess_images([image])[0]
            cur_geo_inputs.append(copy.deepcopy(image))

            _, height, width = image.shape
            if (width // patch_size) % merge_size > 0:
                width -= (width // patch_size) % merge_size * patch_size
            if (height // patch_size) % merge_size > 0:
                height -= (height // patch_size) % merge_size * patch_size

            image_inputs.append(image[:, :height, :width])

        geometry_encoder_inputs.append(torch.stack(cur_geo_inputs))

    return image_inputs, geometry_encoder_inputs


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({args.device}) but no CUDA is available.")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(args.model_path)
    use_geometry_model = getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
    model_class = Qwen2_5_VLForConditionalGenerationWithVGGT if use_geometry_model else Qwen2_5_VLForConditionalGeneration

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_path,
        "config": config,
        "torch_dtype": torch_dtype,
        "device_map": args.device,
    }
    if args.no_flash_attn2:
        model = model_class.from_pretrained(**model_kwargs).eval()
    else:
        model = model_class.from_pretrained(**model_kwargs, attn_implementation="flash_attention_2").eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        padding_side="left",
    )

    visuals = load_visuals(args)
    messages = build_messages(args.prompt, visuals, args.add_frame_index)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, geometry_encoder_inputs = prepare_visual_inputs(messages, processor)

    model_inputs = processor(
        text=text,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
        do_rescale=False,
    )

    if use_geometry_model:
        model_inputs["geometry_encoder_inputs"] = [feat.to(args.device) for feat in geometry_encoder_inputs]
    model_inputs = model_inputs.to(args.device)

    output_ids = model.generate(
        **model_inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, output_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "num_visuals": len(visuals),
        "response": answer,
    }
    print(answer)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
