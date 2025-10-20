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

    model_path = "model_cache/models/Qwen/Qwen3-VL-4B-Instruct"
    image_paths = [
        'demo/images/QC-20230106-0000211_16517.jpeg',
        'demo/images/QC-20230106-0000211_16519.jpeg',
        # "demo/images/QC-202÷30106-0000211_16520.jpeg"
    ]
    prompt = "简要描述这（些）图片。请输出物体的具体坐标。"
    # prompt='Describe the image(s) briefly.'

    if not image_paths:
        raise ValueError("Please set at least one path in image_paths.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            print("Enabled TF32 and cuDNN benchmark for faster inference.")
        except Exception:
            pass

    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    # Inspect template to ensure 图片_ injection is present
    try:
        template_str = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
        if isinstance(template_str, str):
            print(f"Template loaded. Contains '图片_': {'图片_' in template_str}")
        else:
            print("Template string not available on processor.tokenizer; relying on apply_chat_template output.")
    except Exception:
        print("Template inspection failed; proceeding.")

    print(f"Loading model from: {model_path}")
    # Use the specific Qwen3-VL class (AutoModelForCausalLM doesn't map this config)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    # Report dtype support and current model param dtype
    try:
        bf16_ok = getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        print(f"CUDA bf16 supported: {bf16_ok}")
        print(f"Model param dtype: {next(model.parameters()).dtype}")
    except Exception:
        pass
    # Ensure KV cache is enabled for faster decoding
    try:
        model.config.use_cache = True
    except Exception:
        pass
    try:
        model.generation_config.use_cache = True
    except Exception:
        pass
    try:
        attn_impl = getattr(model.config, "attn_implementation", getattr(model.config, "_attn_implementation", None))
        use_cache_flag = getattr(getattr(model, "generation_config", None), "use_cache", None)
        print(f"Attention impl: {attn_impl}; use_cache: {use_cache_flag}")
    except Exception:
        pass

    print("Loading images...")
    images = load_images(image_paths)
    print(f"Loaded {len(images)} image(s):")
    for idx, img in enumerate(images):
        try:
            print(f"  images[{idx}]: size={img.size}")
        except Exception:
            pass

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
    # Preview the templated text and label counts
    preview_len = 600
    preview = text[:preview_len]
    print("\n--- Chat text preview (first 600 chars) ---")
    print(preview)
    print("--- end preview ---\n")
    label_count = text.count("图片_")
    print(f"Occurrences of '图片_': {label_count}; has 图片_1: {'图片_1' in text}, 图片_2: {'图片_2' in text}")

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
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            use_cache=True,
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


