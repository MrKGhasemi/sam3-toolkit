from object_counting import count_objects_video
import os
import numpy as np
import cv2
import torch
from PIL import Image
from config import configs
import models
import utils
import class_generators
import gc
import visualization
import time
from tqdm import tqdm

# Memory Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
except ImportError:
    print("Warning: Could not import SAM 3 box utilities. Geometric prompting might fail.")

# LIMITS:
# If a SINGLE class has more than this many objects on Frame 0, we treat it with caution.
MAX_OBJECTS_PER_CLASS = 40
RESIZE_DIM = 480


def generate_semantic_edges(image_path, args, thickness=2):
    """
    Runs SAM 3 to find objects, then converts their masks into a clean edge map.
    Supports 'blip' and 'llm' modes via model loading.
    Returns output_information dict for visualization.
    """
    print(f"\n--- Generating Semantic Edges for {image_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Image
    try:
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise ValueError("Image not found")
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(image_path).convert('RGB')
        height, width = image_rgb.shape[:2]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {}

    # Determine Class Names (NLP Phase)
    noun_phrases = []

    print(f">> Step 1: Generating classes ({args.mode})...")

    # Load NLP
    blip_model, blip_processor, nlp = models.load_nlp_models(device)

    try:
        if args.mode == "blip":
            temp_models = {
                "blip_model": blip_model,
                "blip_processor": blip_processor,
                "nlp": nlp
            }
            noun_phrases, _ = class_generators.get_classes_blip(
                raw_image, temp_models, device)

        elif args.mode == "llm":
            del blip_model, blip_processor
            torch.cuda.empty_cache()

            temp_models = {"nlp": nlp}
            noun_phrases, _ = class_generators.get_classes_llm(
                True, image_path, temp_models, args.api_key, args.base_url,
                configs.LLM_IMAGE_SEGMENTATION_PARSER_SYSTEM_PROMPT
            )

        # Clean duplicates
        if noun_phrases:
            noun_phrases = list(set([n for n in noun_phrases if n.strip()]))
            print(f"   > Concepts: {noun_phrases}")

    except Exception as e:
        print(f"Error generating classes: {e}")
    finally:
        # Cleanup NLP
        if 'blip_model' in locals():
            del blip_model
        if 'blip_processor' in locals():
            del blip_processor
        if 'nlp' in locals():
            del nlp
        if 'temp_models' in locals():
            del temp_models
        torch.cuda.empty_cache()
        gc.collect()

    if not noun_phrases:
        print("No classes found. Cannot generate edges.")
        return {}

    # SAM Inference (Model Phase)
    print(">> Step 2: Loading SAM 3 for Edge Detection...")
    sam3_model, sam3_processor = models.load_sam3_model(
        device=device, conf_threshold=configs.SAM3_CONF_IMAGE_FOR_COUNTING)

    edge_map = np.zeros((height, width), dtype=np.uint8)
    color_edge_map = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        # Generate colors for classes
        class_colors = visualization.generate_harmonious_colors(
            len(noun_phrases))
        color_dict = {n: class_colors[i] for i, n in enumerate(noun_phrases)}
        class_name_to_id = {n: i for i, n in enumerate(noun_phrases)}

        # Run SAM using helper from utils
        final_masks = utils.process_sam3_segmentation(
            sam3_processor, raw_image, noun_phrases, height, width, class_name_to_id
        )

        if not final_masks:
            print("No masks found for edge detection.")
            return {}

        print(f"Extracting edges from {len(final_masks)} masks...")

        for mask_data in final_masks:
            mask = (mask_data['segmentation'].astype(np.uint8) * 255)

            # Morphological Edge Detection (Outer Gradient)
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

            # Add to BW map
            edge_map = cv2.add(edge_map, edges)

            # Add to Color map
            class_name = mask_data['class_name']
            if class_name in color_dict:
                color = color_dict[class_name]
                color_bgr = (color[2], color[1], color[0])  # RGB to BGR
                color_edge_map[edges > 0] = color_bgr

    finally:
        # Cleanup SAM
        print("   > Unloading SAM 3...")
        del sam3_model
        del sam3_processor
        torch.cuda.empty_cache()
        gc.collect()

    # Pack Output Information
    base_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    output_prefix = os.path.join(args.output, name_without_ext)

    output_information = {
        "edge_bw": edge_map,
        "edge_color_bgr": color_edge_map,
        "output_prefix": output_prefix,
        "mode": args.mode
    }

    return output_information


def semantic_edge_video(video_path, args, thickness=2):
    """
    Wrapper: Uses the robust tracking from object_counting.py, 
    but prepares the output for Edge Visualization.
    """
    # Reuse the existing video tracker
    # uses 'automatic' task_type to auto-detect classes, same as semantic edge did.
    output_information = count_objects_video(
        video_path, args, task_type="automatic")

    if not output_information:
        return {}

    # Inject the specific style parameter for edges
    output_information["thickness"] = thickness

    return output_information
