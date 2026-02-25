import copy
from pycocotools import mask as mask_utils
import pickle
import sys
import torchvision
import ast
import torch
import numpy as np
import cv2
from config import configs
from tqdm import tqdm
from openai import OpenAI
import time
import random

try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
except ImportError:
    print("Warning: Could not import SAM 3 box utilities. Geometric prompting might fail.")

BLOCKED_NOUNS = {
    # Locations / Scene descriptors (Too big to track as objects)
    "space", "area", "zone", "place", "site", "spot", "location",
    "field", "court", "pitch", "ground", "floor", "ceiling", "wall", "grass",
    "dirt", "sand", "beach", "park", "forest", "woods", "mountain", "hill",
    "valley", "sky", "air", "water", "sea", "ocean", "river", "lake", "pond",
    "road", "sidewalk", "path", "harbor", "marina", "highway", "hallway", "corridor",
    "kitchen", "living room", "stadium", "arena", "checkmark",
    "background", "foreground", "scene", "view", "panorama", "landscape", "cleat",
    "nature", "outdoor", "indoor", "outside", "inside", "window", "roof",

    # Abstractions / Actions / Events
    "soccer", "football", "basketball", "tennis", "baseball", "cricket", "rugby",
    "hockey", "golf", "sport", "game", "match", "tournament", "competition", "other",
    "play", "action", "activity", "event", "fun", "time", "day", "night", "settings",
    "light", "shadow", "shade", "reflection", "color", "shape", "texture", "setting",
    "pattern", "design", "style", "art", "work", "job", "way", "direction", "sitting",
    "standing", "sit", "stand", "adult"

    # Image/Camera Terms
    "photo", "image", "picture", "photograph", "snapshot", "shot", "clip", "square",
    "video", "movie", "film", "footage", "frame", "screen", "display", "jersey",
    "close-up", "closeup", "close", "zoom", "focus", "angle", "perspective",
    "view", "shot", "selfie", "portrait", "look", "glance",

    # Generic / Parts / Directions
    "thing", "object", "item", "stuff", "part", "piece", "bit", "portion",
    "segment", "section", "chunk", "fragment", "detail", "feature", "element",
    "component", "side", "front", "back", "top", "bottom", "left", "right",
    "center", "middle", "corner", "edge", "end", "tip", "surface", "line",
    "row", "column", "list", "group", "set", "collection", "batch", "lot",
    "bunch", "pile", "stack", "heap", "couple", "pair", "few", "many",
    "several", "some", "all", "none", "rest", "cloth", "short", "foliage",
    "bird", "jean", "sneaker", "pavement",

    # Body Parts
    "hand", "foot", "feet", "leg", "arm", "head", "face", "body", "hair",
    "eye", "ear", "nose", "mouth", "lip", "finger", "toe", "skin", "knee",
    "elbow", "shoulder", "back", "neck"
}
PERSON_REPLACE_LIST = ["individual", "pedestrian", 'man', 'woman', 'child',
                       'boy', 'girl', 'people', 'he', 'she', 'they']


def filter_synonyms(nouns, nlp, threshold=None):
    """
    Filters a list of nouns, removing synonyms based on vector similarity.
    """
    if threshold is None:
        threshold = configs.SYNONYM_THRESHOLD

    noun_tokens = [nlp(noun) for noun in nouns]
    final_nouns = []

    for i in range(len(noun_tokens)):
        token_a = noun_tokens[i]

        if token_a.vector_norm == 0:
            if token_a.text not in final_nouns:
                final_nouns.append(token_a.text)
            continue

        is_synonym = False
        for j in range(len(final_nouns)):
            token_b = nlp(final_nouns[j])
            if token_b.vector_norm == 0:
                continue

            similarity = token_a.similarity(token_b)
            if similarity > threshold:
                is_synonym = True
                print(
                    f"--- Filtering '{token_a.text}' (similar to '{token_b.text}', score: {similarity:.2f})")
                break

        if not is_synonym:
            final_nouns.append(token_a.text)

    return final_nouns


def normalize_person_classes(nouns):
    """
    Merges all person-related nouns into a single 'person' class.
    """
    replace_list = PERSON_REPLACE_LIST
    final_nouns = []
    person_added = False
    for noun in nouns:
        if noun in replace_list:
            if not person_added:
                final_nouns.append('person')
                person_added = True
        else:
            final_nouns.append(noun)
    return final_nouns


def parse_and_clean_nouns(raw_text, nlp):
    """
    Full NLP pipeline: parses text, extracts nouns, filters, and normalizes.
    """
    # Lowercase for consistent matching and Extract head nouns
    doc = nlp(raw_text.lower())
    all_noun_phrases = []

    for chunk in doc.noun_chunks:
        head = chunk.root.lemma_.lower()

        # Basic Validity Check
        if not head.isalpha():
            continue

        # Blocklist Filter
        if head in BLOCKED_NOUNS:
            continue

        # Stopword Filter
        if head in {"a", "an", "the", "this", "that", "it", "they"}:
            continue

        all_noun_phrases.append(head)

    # Simple de-duplication
    seen = set()
    raw_noun_list = [x for x in all_noun_phrases if not (
        x in seen or seen.add(x))]

    # Semantic de-duplication
    print(f"Raw nouns (pre-filter): {raw_noun_list}")
    filtered_nouns = filter_synonyms(raw_noun_list, nlp)

    # Normalize person classes (man/woman -> person)
    final_nouns = normalize_person_classes(filtered_nouns)

    print(f"Parsed clean object nouns: {final_nouns}")
    return final_nouns


def calculate_box_iou_internal(box1, box2):
    """
    Calculates IoU between two [x1, y1, x2, y2] boxes.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return interArea / float(box1Area + box2Area - interArea)


def resize_video_if_needed(input_path, output_path, max_dim=480):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max(width, height) <= max_dim:
        print(
            f"   [Info] Video dimensions {width}x{height} are within limit {max_dim}.")

    print(
        f"   [Memory Opt] Resizing video from {width}x{height} to max_dim={max_dim}...")
    scale = max_dim / max(width, height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    for _ in tqdm(range(total_frames), desc="Resizing Video"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(frame_resized)

    cap.release()
    out.release()
    return True


def parse_sam3_result(out_data, frame_label="?"):
    frame_objs = []
    obj_ids = out_data.get("out_obj_ids", out_data.get("obj_ids", []))

    masks_tensor = None
    for k in ["out_binary_masks", "mask", "masks", "segmentation"]:
        if k in out_data:
            masks_tensor = out_data[k]
            break

    if masks_tensor is not None and len(obj_ids) > 0:
        if isinstance(masks_tensor, torch.Tensor):
            masks_np = (masks_tensor > 0.0).float().cpu().numpy()
        else:
            masks_np = masks_tensor

        for i, obj_id in enumerate(obj_ids):
            try:
                if masks_np.ndim == 4:
                    single_mask = masks_np[i, 0]
                elif masks_np.ndim == 3:
                    single_mask = masks_np[i]
                else:
                    continue

                frame_objs.append({
                    'obj_id': int(obj_id),
                    'mask': single_mask.astype(bool)
                })
            except IndexError:
                continue
    return frame_objs


def search_engine_clean_nouns(raw_text, api_key, base_url, system_prompt):
    """
    Parses text to extract objects using LLM and returns a Python List.
    """
    print(f"Generating clean_nouns with external LLM...")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    if "gemini" in configs.LLM_PARSER_MODEL_NAME:
        temperature = 0.2
    else:
        temperature = 1.0

    current_time = str(time.time())
    random_seed = str(random.randint(0, 10000))

    classes_str = "[]"  # Default safety

    try:
        respond = client.chat.completions.create(
            model=configs.LLM_PARSER_MODEL_NAME,
            messages=[
                {"role": "system",
                    "content": f"{system_prompt}. [Ref: {current_time}-{random_seed}]. If you cannot see the classes, reply with '[]' or 'NO_CLASSES'."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"""Here is a raw list of object nouns and phrases: {raw_text}.
                                                Clean this list according to the guidelines.
                                                Ensure all distinct physical objects are represented without missing any, but remove all messy repetitions and non-object noise.
                                                Result (return ONLY a Python list of strings):"""}
                ]}
            ],
            temperature=temperature,
            max_tokens=14000
        )

        classes_str = respond.choices[0].message.content
        print(f"LLM Noun Parser Response: '{classes_str}'")

        # Handle "NO_CLASSES" or empty responses
        if "NO_CLASSES" in classes_str or not classes_str.strip():
            return []

        # 2. Clean markdown formatting (e.g. ```json ... ```)
        cleaned_str = classes_str.strip()
        if cleaned_str.startswith("```"):
            cleaned_str = cleaned_str.split("```")[1]
            if cleaned_str.startswith("json"):
                cleaned_str = cleaned_str[4:]
            elif cleaned_str.startswith("python"):
                cleaned_str = cleaned_str[6:]

        cleaned_str = cleaned_str.strip()

        # Safely convert String -> List
        try:
            final_list = ast.literal_eval(cleaned_str)
            if isinstance(final_list, list):
                return final_list
        except Exception:
            return [x.strip().strip('"').strip("'") for x in cleaned_str.strip('[]').split(',') if x.strip()]

    except Exception as e:
        print(f"Error calling LLM API for Parsing: {e}")
        return []

    return []


def process_sam3_segmentation(processor, raw_image, noun_phrases, height, width, class_name_to_id):
    """
    Helper function to run SAM3 on a list of nouns and return filtered masks.
    """
    inference_state = processor.set_image(raw_image)
    detected_masks = []

    for noun in noun_phrases:
        processor.reset_all_prompts(inference_state)
        print(f"  > Scanning for: '{noun}'...")

        try:
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=noun)

            if 'masks' not in inference_state or inference_state['masks'] is None:
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

                if score < configs.SAM3_CONF_IMAGE_FOR_COUNTING:
                    continue

                y_indices, x_indices = np.where(mask_bool)
                if len(y_indices) == 0:
                    continue
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                detected_masks.append({
                    'segmentation': mask_bool,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'area': int(mask_bool.sum()),
                    'score': score,
                    'class_name': noun,
                    'class_id': class_name_to_id.get(noun, 0)
                })

        except Exception as e:
            print(f"    Error processing '{noun}': {e}")
            continue

    if not detected_masks:
        return []

    boxes_tensor = torch.tensor([m['bbox']
                                for m in detected_masks], dtype=torch.float32)
    scores_tensor = torch.tensor([m['score']
                                 for m in detected_masks], dtype=torch.float32)
    keep_indices = torchvision.ops.nms(
        boxes_tensor, scores_tensor, iou_threshold=configs.SAM3_IMAGE_SEGMENTATION_NMS_THRESHOLD)

    return [detected_masks[i] for i in keep_indices.tolist()]


def compress_and_save_output_information(output_info, filename):
    """
    Compresses binary masks to RLE format and saves to pickle.
    """
    print("--- Compressing masks...")

    # Deep copy to avoid modifying the original data in memory if you are still using it
    compressed_info = copy.deepcopy(output_info)

    if "inference_results" in compressed_info:
        for frame_idx, objects in compressed_info["inference_results"].items():
            for obj in objects:
                if "mask" in obj:
                    # RLE requires Fortran-contiguous array
                    mask_np = np.asfortranarray(obj["mask"].astype(np.uint8))
                    # Encode
                    obj["mask"] = mask_utils.encode(mask_np)

    print(f"--- Saving compressed data to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(compressed_info, f)
    print("Save complete.")


def load_and_decompress_output_information(filename):
    """
    Loads pickle and decodes RLE masks back to boolean numpy arrays.
    """
    print(f"--- Loading from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print("--- Decompressing masks...")
    if "inference_results" in data:
        for frame_idx, objects in data["inference_results"].items():
            for obj in objects:
                if "mask" in obj:
                    # Check if it is actually RLE (it will be a dict or bytes)
                    if isinstance(obj["mask"], dict) or isinstance(obj["mask"], bytes):
                        # Decode and convert back to boolean
                        obj["mask"] = mask_utils.decode(
                            obj["mask"]).astype(bool)

    print("Load complete.")
    return data
