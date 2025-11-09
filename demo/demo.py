from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.config.prompts import (
    SYSTEM_PROMPT,  # noqa
    SYSTEM_PROMPT_SUMMARY,  # noqa
)


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

    # model_path = "output/summary_merged/10-25-aug_on-full_last2_llm"
    model_path = "output/stage_2_gkd_merged/11-08/checkpoint-200"

    image_paths = [
        # "demo/images/QC-20230106-0000211_16517.jpeg",
        # "demo/images/QC-20230106-0000211_16519.jpeg",
        # "demo/images/test_demo.jpg",
        # ]
        # "demo/irrelevant_images/QC-TEMP-20241028-0015135_4206555.jpeg",
        # "demo/irrelevant_images/QC-TEMP-20241028-0015135_4206556.jpeg",
        # "demo/irrelevant_images/QC-TEMP-20241028-0015135_4206715.jpeg",
    ]
    prompt = "请描述这张图片"
    max_new_tokens = 512
    temperature = 0.0
    top_p = 0.9
    top_k = None
    repetition_penalty = 1.05
    # prompt='Describe the image(s) briefly.'

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print("Enabled TF32 and cuDNN benchmark for faster inference.")

    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    # Inspect template availability (optional informational output)
    try:
        template_str = getattr(
            getattr(processor, "tokenizer", None), "chat_template", None
        )
        if isinstance(template_str, str):
            print(f"Template loaded (length={len(template_str)} characters).")
        else:
            print(
                "Template string not available on processor.tokenizer; relying on apply_chat_template output."
            )
    except Exception:
        print("Template inspection failed; proceeding.")

    print(f"Loading model from: {model_path}")
    # Use the specific Qwen3-VL class (AutoModelForCausalLM doesn't map this config)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(torch.device(device))  # type: ignore[arg-type]
    model.eval()
    # Report dtype support and current model param dtype
    try:
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
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
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            gc.use_cache = True
    except Exception:
        pass
    try:
        attn_impl = getattr(
            model.config,
            "attn_implementation",
            getattr(model.config, "_attn_implementation", None),
        )
        use_cache_flag = getattr(
            getattr(model, "generation_config", None), "use_cache", None
        )
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
        # {
        #     "role": "system",
        #     "content": SYSTEM_PROMPT_SUMMARY,
        # },
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
    print("\n--- FULL Chat text (for debugging) ---")
    print(text)
    print("--- end full text ---\n")
    print(f"Prompt length: {len(text)} characters")

    print("Preprocessing inputs...")
    if images:
        inputs = processor(images=images, text=text, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")

    # Optional: debug shapes
    show_input_debug(inputs)

    # Move tensors to device
    dev_obj = torch.device(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(dev_obj)

    print("Generating...")
    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
        "do_sample": do_sample,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # Trim input prefix to get only newly generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
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
