import os
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.config.prompts import SYSTEM_PROMPT_B


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(img)
    return images


def show_input_debug(inputs):
    def shape_str(x):
        return f"shape={tuple(x.shape)} dtype={x.dtype}"

    for k in [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    ]:
        if k in inputs and inputs[k] is not None:
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                print(f"{k}: {shape_str(v)}")
            else:
                print(f"{k}: type={type(v)}")


def main() -> None:
    # Configuration (edit these)
    checkpoint_path = "output/standard/v0-20251019-073512/checkpoint-1098"
    image_paths = [
        'demo/images/QC-20230106-0000211_16517.jpeg',
        'demo/images/QC-20230106-0000211_16519.jpeg',
    ]
    prompt = "简要描述这（些）图片。请输出物体的具体坐标。"
    # prompt='Describe the image(s) briefly.'

    if not image_paths:
        raise ValueError("Please set at least one path in image_paths.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor from: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    print(f"Loading model from: {checkpoint_path}")
    # Use the specific Qwen3-VL class (AutoModelForCausalLM doesn't map this config)
    model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint_path, torch_dtype="auto")
    model.to(device)
    model.eval()

    print("Loading images...")
    images = load_images(image_paths)

    # Build chat-style messages with system prompt and user turn with images then text prompt
    message_content = [{"type": "image", "image": img} for img in images]
    message_content.append({"type": "text", "text": prompt})
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_B,
        },
        {
            "role": "user",
            "content": message_content,
        },
    ]

    # Create text with multimodal template, then preprocess vision + text together
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Preprocessing inputs...")
    inputs = processor(
        images=images,
        text=text,
        return_tensors="pt",
    )

    # Optional: debug shapes
    show_input_debug(inputs)

    # Move tensors to device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    print("Generating...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8000,
            do_sample=False,
        )

    # Trim input prefix to get only newly generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Decode
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print("\n=== Model response ===")
    print(outputs[0] if outputs else "")


if __name__ == "__main__":
    main()


