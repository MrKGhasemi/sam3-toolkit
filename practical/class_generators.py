from openai import AsyncOpenAI
import asyncio
import torch
import utils
from config import configs
from PIL import Image
import io
import base64
import time
import random
import visualization


def get_classes_blip(raw_image, models, device="cuda"):
    """
    Generates noun list using BLIP-Large captioning on 9 image crops.
    """
    blip_model = models['blip_model']
    blip_processor = models['blip_processor']
    nlp = models['nlp']

    images = visualization.get_image_crops(raw_image)
    all_captions = []

    print(f"Generating {len(images)} captions with BLIP-Large...")
    for i, img_crop in enumerate(images):
        blip_inputs = blip_processor(
            images=img_crop, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            blip_out = blip_model.generate(
                **blip_inputs, max_new_tokens=configs.BLIP_MAX_TOKENS)

        caption_text = blip_processor.decode(
            blip_out[0], skip_special_tokens=True)
        print(f"BLIP Caption {i}: '{caption_text}'")
        all_captions.append(caption_text)

    full_caption_text = " . ".join(all_captions)
    clean_nouns = utils.parse_and_clean_nouns(full_caption_text, nlp)
    text_prompt = ". ".join(clean_nouns)

    return clean_nouns, text_prompt


async def process_single_crop(client, img_crop, index, system_prompt, user_prompt, model_name):
    try:
        # Prepare Image
        img_crop.thumbnail((512, 512))
        buffered = io.BytesIO()
        img_crop.save(buffered, format="JPEG", quality=75)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Setup Params
        current_time = str(time.time())
        random_seed = str(random.randint(0, 10000))
        temperature = 0.2 if "gemini" in model_name else 1.0

        # Async API Call
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                    "content": f"{system_prompt} [Ref: {current_time}-{random_seed}]"},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": image_url, "detail": "low"}}
                ]}
            ],
            temperature=temperature,
            max_tokens=500
        )

        caption = response.choices[0].message.content
        if "NO_IMAGE" in caption:
            return ""

        print(f"LLM Response (Crop {index}): '{caption}'")
        return caption.replace(",", ". ")

    except Exception as e:
        print(f"Error on Crop {index}: {e}")
        return ""


def get_classes_llm(search_task, image_path, models, api_key, base_url, system_prompt):
    nlp = models['nlp']

    # Load Image
    try:
        if isinstance(image_path, str):
            raw_image = Image.open(image_path)
        else:
            raw_image = image_path
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return [], ""

    # Get Crops
    images = visualization.get_image_crops(raw_image)
    print(
        f"Generating classes with Async LLM ({len(images)} crops parallel)...")

    # Define the Async Loop Logic
    async def run_parallel_captions():
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        tasks = []

        for i, img_crop in enumerate(images):
            task = process_single_crop(
                client, img_crop, i,
                configs.LLM_CAPTION_SYSTEM_PROMPT,
                configs.LLM_CAPTION_USER_PROMPT,
                configs.LLM_CAPTION_MODEL_NAME
            )
            tasks.append(task)

        # Wait for ALL crops to finish (Parallel Execution)
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    # Execute Async Loop Synchronously
    # This enables "await" logic without changing your entire codebase structure
    try:
        try:
            # If running in a script
            all_captions = asyncio.run(run_parallel_captions())
        except RuntimeError:
            # If running in Jupyter/Colab (Event loop already exists)
            import nest_asyncio
            nest_asyncio.apply()
            all_captions = asyncio.get_event_loop().run_until_complete(run_parallel_captions())

    except Exception as e:
        print(f"Async Loop Failed: {e}")
        return [], ""

    if not all_captions:
        print("Error: No captions generated.")
        return [], ""

    # Clean Nouns (Sequential Step - Clean AFTER all crops are done)
    full_caption_text = " . ".join(all_captions)

    if search_task:
        clean_nouns = utils.search_engine_clean_nouns(
            full_caption_text, api_key, base_url, system_prompt)
    else:
        clean_nouns = utils.parse_and_clean_nouns(full_caption_text, nlp)

    text_prompt = ". ".join(clean_nouns)
    return clean_nouns, text_prompt
