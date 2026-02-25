import visualization
import gc
import torchvision
import class_generators
import models
from config import configs
from PIL import Image
import torch
import cv2
import numpy as np
import os

try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
except ImportError:
    print("Warning: Could not import SAM 3 box utilities. Geometric prompting might fail.")


def process_image(image_path, args, save_viz=False):
    """
    Main pipeline for General Segmentation (Image Mode).
    Loads NLP models -> Generates Classes -> Unloads NLP -> Loads SAM3 -> Segments -> Unloads SAM3.
    Returns a dictionary of information for visualization.
    """
    print(f"\n--- Processing {image_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image_data = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(image_path).convert('RGB')
        height, width = image_rgb.shape[:2]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {"masks": [], "image_rgb": None}

    # Determine Class Names (NLP Phase)
    noun_phrases = ["object"]

    if args.mode == "blip":
        print(">> Step 1: Loading NLP models for BLIP captioning...")
        blip_model, blip_processor, nlp = models.load_nlp_models(device)

        temp_models_dict = {
            "blip_model": blip_model,
            "blip_processor": blip_processor,
            "nlp": nlp
        }

        try:
            noun_phrases, _ = class_generators.get_classes_blip(
                raw_image, temp_models_dict, device
            )
        except Exception as e:
            print(f"Error in BLIP generation: {e}")
        finally:
            print("   > Unloading NLP models...")
            del blip_model
            del blip_processor
            del nlp
            del temp_models_dict
            torch.cuda.empty_cache()
            gc.collect()

    elif args.mode == "llm":
        # LLM mode still needs Spacy (nlp) for cleaning nouns
        print(">> Step 1: Loading NLP (Spacy) for LLM cleaning...")

        blip_model, blip_processor, nlp = models.load_nlp_models(device)

        # save memory
        del blip_model
        del blip_processor
        torch.cuda.empty_cache()

        temp_models_dict = {"nlp": nlp}

        try:
            # LLM Parallel Crops & Parsing by passing True
            noun_phrases, _ = class_generators.get_classes_llm(
                True, image_path, temp_models_dict, args.api_key, args.base_url,
                configs.LLM_IMAGE_SEGMENTATION_PARSER_SYSTEM_PROMPT
            )
        except Exception as e:
            print(f"Error in LLM generation: {e}")
        finally:
            print("   > Unloading Spacy...")
            del nlp
            del temp_models_dict
            gc.collect()

    if not noun_phrases:
        print("Warning: No class names found. Skipping image.")
        return {"masks": [], "image_rgb": image_rgb}

    # Clean duplicates
    noun_phrases = list(set([n for n in noun_phrases if n.strip()]))
    print(f"Detected Concepts ({len(noun_phrases)}): {noun_phrases}")

    # Setup Colors (Randomized)
    # generate_harmonious_colors is now random each call
    CLASS_COLORS = visualization.generate_harmonious_colors(len(noun_phrases))
    noun_phrases_with_bg = ["background"] + noun_phrases
    CLASS_COLORS_with_bg = [(0, 0, 0)] + CLASS_COLORS
    CLASS_NAME_TO_ID = {name: i for i, name in enumerate(noun_phrases_with_bg)}

    # SAM3 Prediction (Load SAM3 Model)
    print(">> Step 2: Loading SAM 3 Image Model...")
    sam3_model, sam3_processor = models.load_sam3_model(
        device=device, conf_threshold=configs.SAM3_CONF_IMAGE_THRESHOLD)

    final_masks = []

    try:
        inference_state = sam3_processor.set_image(raw_image)
        all_masks = []

        # Loop through each noun
        for i, noun in enumerate(noun_phrases):
            print(f"  > Prompting for: '{noun}'...")
            sam3_processor.reset_all_prompts(inference_state)

            try:
                inference_state = sam3_processor.set_text_prompt(
                    state=inference_state,
                    prompt=noun
                )

                if 'masks' not in inference_state:
                    continue

                pred_masks = inference_state['masks']
                pred_scores = inference_state['scores']

                if isinstance(pred_masks, torch.Tensor):
                    pred_masks = pred_masks.reshape(-1, height, width)
                    pred_scores = pred_scores.reshape(-1)

                if pred_masks.dtype == torch.bool:
                    masks_np = pred_masks.cpu().numpy()
                else:
                    masks_np = (pred_masks > 0.0).float().cpu().numpy()

                scores_np = pred_scores.cpu().numpy()

                for j in range(len(masks_np)):
                    mask_bool = masks_np[j].astype(bool)
                    score = float(scores_np[j])

                    if score < configs.SAM3_CONF_IMAGE_THRESHOLD:
                        continue

                    y_indices, x_indices = np.where(mask_bool)
                    if len(y_indices) == 0:
                        continue

                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bbox = [x_min, y_min, x_max, y_max]

                    all_masks.append({
                        'segmentation': mask_bool,
                        'bbox': bbox,
                        'area': int(mask_bool.sum()),
                        'score': score,
                        'class_name': noun,
                        'class_id': CLASS_NAME_TO_ID[noun]
                    })

            except Exception as e:
                print(f"    Error prompting for '{noun}': {e}")
                continue

        # NMS
        if all_masks:
            print(f"Aggregated {len(all_masks)} masks. Running NMS...")
            boxes_tensor = torch.tensor(
                [m['bbox'] for m in all_masks], dtype=torch.float32)
            scores_tensor = torch.tensor(
                [m['score'] for m in all_masks], dtype=torch.float32)
            keep_indices = torchvision.ops.nms(
                boxes_tensor, scores_tensor, iou_threshold=configs.SAM3_IMAGE_SEGMENTATION_NMS_THRESHOLD)
            final_masks = [all_masks[i] for i in keep_indices.tolist()]
            print(f"Final Count: {len(final_masks)} unique objects.")
        else:
            print("No masks found.")

    finally:
        # Cleanup SAM3
        print("   > Unloading SAM 3 model...")
        del sam3_model
        del sam3_processor
        del inference_state
        torch.cuda.empty_cache()
        gc.collect()

    # Pack Output Information
    output_prefix = None
    if final_masks:
        base_filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        output_prefix = os.path.join(args.output, name_without_ext)

    output_information = {
        "masks": final_masks,
        "image_rgb": image_rgb,
        "class_colors": CLASS_COLORS_with_bg,
        "output_prefix": output_prefix,
        "mode": args.mode
    }

    if save_viz and final_masks:
        print("Saving visualization directly from process_image...")
        visualization.show_save_visualizations(
            output_information, figsize=(20, 10))

    return output_information
