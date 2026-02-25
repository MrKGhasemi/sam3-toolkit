import visualization
import time
import gc
import traceback
import torchvision
import class_generators
import utils
import models
from config import configs
from PIL import Image
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
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


def count_objects(image_path, args, task_type="automatic", prompt_value=None):
    """
    Main pipeline for Counting (Image Mode).
    Loads NLP (if automatic) -> Unload -> Loads SAM3 -> Count -> Unload.
    Returns: output_information (dict)
    """
    print(f"\n--- Starting Count Task: {task_type} on {image_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image_data = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(image_path).convert('RGB')
        height, width = image_rgb.shape[:2]
        print(f"Image Dimensions: {width}x{height}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return {}

    # Determine Prompts (NLP Phase)
    target_nouns = []

    if task_type == "automatic":
        print(f">> Auto-generating prompt ({args.mode})...")

        # Load NLP Models
        blip_model, blip_processor, nlp = models.load_nlp_models(device)

        try:
            if args.mode == "blip":
                temp_models = {
                    "blip_model": blip_model,
                    "blip_processor": blip_processor,
                    "nlp": nlp
                }
                target_nouns, _ = class_generators.get_classes_blip(
                    raw_image, temp_models, device)

            elif args.mode == "llm":
                # save memory
                del blip_model, blip_processor
                torch.cuda.empty_cache()

                temp_models = {"nlp": nlp}

                target_nouns, _ = class_generators.get_classes_llm(
                    True, image_path, temp_models, args.api_key, args.base_url,
                    configs.LLM_IMAGE_SEGMENTATION_PARSER_SYSTEM_PROMPT
                )

            # Cleanup duplicates
            if target_nouns:
                target_nouns = list(
                    set([n for n in target_nouns if n.strip()]))

        except Exception as e:
            print(f"Error generating classes: {e}")
        finally:
            print("   > Unloading NLP models...")
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

    elif task_type == "prompt":
        print(f">> Task is prompt for: {prompt_value}...")

        if not prompt_value or not isinstance(prompt_value, str):
            print("Error: 'name' mode requires a string prompt.")
            return {}
        # Parse comma-separated string into list
        target_nouns = [n.strip()
                        for n in prompt_value.split(',') if n.strip()]

    # SAM3 Processing (Model Phase)
    print(">> Loading SAM 3 Image Model...")
    sam3_model, sam3_processor = models.load_sam3_model(
        device=device, conf_threshold=configs.SAM3_CONF_IMAGE_FOR_COUNTING)

    final_masks = []
    input_prompt_map = None

    try:
        if task_type == "automatic" or task_type == "prompt":
            if not target_nouns:
                print("No target nouns found/provided.")
            else:
                class_name_to_id = {name: i for i,
                                    name in enumerate(target_nouns)}

                final_masks = utils.process_sam3_segmentation(
                    sam3_processor, raw_image, target_nouns, height, width, class_name_to_id)

                if task_type == "automatic":
                    # Filter singletons for automatic mode
                    counts = {}
                    for m in final_masks:
                        counts[m['class_name']] = counts.get(
                            m['class_name'], 0) + 1
                    final_masks = [
                        m for m in final_masks if counts[m['class_name']] >= 2]

        elif task_type == "geometric":
            print(f"Using SAM 3 Geometric Prompt (Hybrid Mode)...")
            input_prompt_map = visualization.draw_geometric_inputs(
                image_rgb, prompt_value)

            # Parse Inputs
            pos_prompts = []
            neg_prompts = []
            text_prompt = None

            if isinstance(prompt_value, dict):
                text_prompt = prompt_value.get('text', None)
                pos_prompts = prompt_value.get('positive', [])
                neg_prompts = prompt_value.get('negative', [])
                if 'box' in prompt_value:
                    pos_prompts.append(prompt_value['box'])
                if 'point' in prompt_value:
                    pos_prompts.append(prompt_value['point'])
            elif isinstance(prompt_value, (list, tuple, np.ndarray)):
                if len(prompt_value) > 0 and isinstance(prompt_value[0], (list, tuple, np.ndarray)):
                    pos_prompts = prompt_value
                else:
                    pos_prompts = [prompt_value]

            # Setup Inference
            inference_state = sam3_processor.set_image(raw_image)
            sam3_processor.reset_all_prompts(inference_state)

            if text_prompt:
                print(f"  > Applying Text Prompt: '{text_prompt}'")
                inference_state = sam3_processor.set_text_prompt(
                    state=inference_state, prompt=text_prompt)

            # Geometric Prompts
            valid_boxes_count = 0

            def add_prompts(prompts, is_positive):
                count = 0
                for p in prompts:
                    flat_input = list(map(float, p))
                    if len(flat_input) == 4:
                        x1, y1, x2, y2 = flat_input
                        if x1 > width or y1 > height:
                            continue
                        w_b, h_b = x2 - x1, y2 - y1
                        box_xywh = torch.tensor([x1, y1, w_b, h_b]).view(-1, 4)
                    elif len(flat_input) == 2:
                        px, py = flat_input
                        if px > width or py > height:
                            continue
                        box_xywh = torch.tensor(
                            [px - 2, py - 2, 4.0, 4.0]).view(-1, 4)
                    else:
                        continue

                    box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                    norm_box = normalize_bbox(
                        box_cxcywh, width, height).flatten().tolist()
                    sam3_processor.add_geometric_prompt(
                        state=inference_state, box=norm_box, label=is_positive
                    )
                    count += 1
                return count

            if pos_prompts:
                valid_boxes_count += add_prompts(pos_prompts, True)
            if neg_prompts:
                add_prompts(neg_prompts, False)

            if not text_prompt and valid_boxes_count == 0:
                print("Error: No valid text or geometric prompts.")
            else:
                # Get Results
                if 'masks' in inference_state and inference_state['masks'] is not None:
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

                    target_name = text_prompt if text_prompt else "target"

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

                        final_masks.append({
                            'segmentation': mask_bool,
                            'bbox': [x_min, y_min, x_max, y_max],
                            'area': int(mask_bool.sum()),
                            'score': score,
                            'class_name': target_name,
                            'class_id': 0
                        })
    except Exception as e:
        print(f"Error during SAM3 processing: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        print("   > Unloading SAM 3 model...")
        del sam3_model
        del sam3_processor
        if 'inference_state' in locals():
            del inference_state
        torch.cuda.empty_cache()
        gc.collect()

    if not final_masks:
        print("No objects found.")
        return {}

    # NMS for Geometric Mode
    if task_type == "geometric":
        boxes_tensor = torch.tensor([m['bbox']
                                    for m in final_masks], dtype=torch.float32)
        scores_tensor = torch.tensor(
            [m['score'] for m in final_masks], dtype=torch.float32)
        keep_indices = torchvision.ops.nms(
            boxes_tensor, scores_tensor, iou_threshold=configs.SAM3_IMAGE_SEGMENTATION_NMS_THRESHOLD)
        final_masks = [final_masks[i] for i in keep_indices.tolist()]

    print(f"Total counted objects: {len(final_masks)}")

    # Pack Output Information
    base_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    output_prefix = os.path.join(
        args.output, f"{name_without_ext}_{task_type}")

    # Identify unique classes found to help with color generation later
    unique_classes = list(set([m['class_name'] for m in final_masks]))
    unique_classes.sort()

    output_information = {
        "masks": final_masks,
        "image_rgb": image_rgb,
        "input_map": input_prompt_map,
        "output_prefix": output_prefix,
        "mode": args.mode,
        "task_type": task_type,
        "found_classes": unique_classes
    }

    return output_information


def count_objects_video(video_path, args, task_type="automatic", prompt_value=None):
    """
    Counts objects in video. Returns output_information dict.
    """
    print(f"\n--- Starting VIDEO Count: {task_type} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # video prep
    if not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        return {}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   > Video Path: {video_path}")
    print(f"   > Total Frames: {total_frames}")

    # capture frame 0
    ret, frame0 = cap.read()
    cap.release()

    if not ret:
        print("Could not read Frame 0.")
        return {}

    # show frame 0
    print("   > Visualizing Frame 0:")
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Frame 0")
    plt.show()

    # determine prompts
    target_nouns = []

    if task_type == "automatic":
        print(f">> Generating classes ({args.mode})...")

        if args.mode == "blip":
            blip_model, blip_processor, nlp = models.load_nlp_models(device)
            try:
                image_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                from PIL import Image
                generated_nouns, _ = class_generators.get_classes_blip(
                    Image.fromarray(image_rgb),
                    {"blip_model": blip_model,
                        "blip_processor": blip_processor, "nlp": nlp},
                    device
                )
                target_nouns = list(
                    set([n for n in generated_nouns if n.strip()]))
            finally:
                del blip_model, blip_processor, nlp
                torch.cuda.empty_cache()
                gc.collect()

        elif args.mode == "llm":
            print("   > Loading Spacy...")
            _bm, _bp, nlp = models.load_nlp_models(device)
            del _bm, _bp
            torch.cuda.empty_cache()

            try:
                unique_name = f"debug_frame_{int(time.time())}.jpg"
                temp_frame_path = os.path.join(args.output, unique_name)
                cv2.imwrite(temp_frame_path, frame0)

                generated_nouns, _ = class_generators.get_classes_llm(
                    True,
                    temp_frame_path,
                    {"nlp": nlp},
                    args.api_key,
                    args.base_url,
                    configs.LLM_IMAGE_SEGMENTATION_PARSER_SYSTEM_PROMPT
                )

                target_nouns = list(
                    set([n for n in generated_nouns if n.strip()]))
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            except Exception as e:
                print(f"   > LLM Error: {e}")
            finally:
                del nlp
                gc.collect()

        print(f"   > Classes found: {target_nouns}")

    elif task_type == "prompt":
        print(f">> Task is prompt for: {prompt_value}...")

        if isinstance(prompt_value, str):
            target_nouns = [n.strip() for n in prompt_value.split(',')]
        else:
            target_nouns = []

    if not target_nouns:
        print("   > No classes found. Exiting.")
        return {}

    # Initialize SAM3 and Track
    print(">> Initializing SAM 3...")
    predictor = models.load_sam3_video_model(device)
    final_all_frames_results = {}
    id_offset = 0

    # Output Prefix Generation
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    final_prefix = os.path.join(args.output, f"{video_name}_{task_type}")

    for noun in target_nouns:
        print(f"\n   > Processing Class: '{noun}'")
        session_id = None
        try:
            response = predictor.handle_request(request=dict(
                type="start_session", resource_path=video_path
            ))
            session_id = response["session_id"]

            # Dry run frame 0
            with torch.inference_mode():
                resp = predictor.handle_request(request=dict(
                    type="add_prompt", session_id=session_id, frame_index=0, text=noun
                ))

            initial_objs = utils.parse_sam3_result(resp["outputs"], "0")
            if not initial_objs:
                print(f"     -> No '{noun}' found on Frame 0.")
                continue

            print(
                f"     -> Tracking {len(initial_objs)} instances from frame 0, possibly find more in forward frames...")

            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                stream = predictor.handle_stream_request(request=dict(
                    type="propagate_in_video", session_id=session_id
                ))

                for r in tqdm(stream, total=total_frames, leave=False):
                    f_idx = r["frame_index"]
                    out_data = r["outputs"]

                    # Manual parsing to filter by SCORE
                    masks_tensor = out_data.get("out_binary_masks")
                    scores_tensor = out_data.get("out_obj_scores")
                    obj_ids_tensor = out_data.get("out_obj_ids")

                    valid_objs = []

                    if masks_tensor is not None and obj_ids_tensor is not None:
                        # Normalize to numpy/cpu
                        if isinstance(masks_tensor, torch.Tensor):
                            masks_np = (masks_tensor >
                                        0.0).float().cpu().numpy()
                            obj_ids_np = obj_ids_tensor.cpu().numpy()
                            if scores_tensor is not None:
                                scores_np = scores_tensor.flatten().cpu().numpy()
                            else:
                                scores_np = None
                        else:
                            masks_np = masks_tensor
                            obj_ids_np = obj_ids_tensor
                            scores_np = scores_tensor.flatten() if scores_tensor is not None else None

                        # Iterate through objects in this frame
                        for i in range(len(obj_ids_np)):
                            # Check Confidence Threshold
                            if scores_np is not None:
                                if scores_np[i] < configs.SAM3_CONF_OBJECT_COUNTING_VIDEO_THRESHOLD:
                                    continue

                            # Check Mask Validity
                            m = masks_np[i]
                            if m.ndim == 3:
                                m = m[0]
                            m_bool = m.astype(bool)

                            # Filter small noise
                            if m_bool.sum() <= 50:
                                continue

                            # BBox Calculation
                            y_idx, x_idx = np.where(m_bool)
                            if len(y_idx) > 0:
                                x1, x2 = x_idx.min(), x_idx.max()
                                y1, y2 = y_idx.min(), y_idx.max()
                                bbox = [x1, y1, x2, y2]
                            else:
                                bbox = [0, 0, 0, 0]

                            valid_objs.append({
                                'obj_id': int(obj_ids_np[i]),
                                'mask': m_bool,
                                'bbox': bbox,
                                # Placeholder for global
                                'original_id': int(obj_ids_np[i]),
                                'class_name': noun
                            })

                    # Add to Global Results if any valid objects found
                    if valid_objs:
                        for obj in valid_objs:
                            # Apply offset to distinguish between different class runs
                            obj['original_id'] = obj['obj_id']
                            obj['obj_id'] = obj['obj_id'] + id_offset
                            obj['class_name'] = noun

                        if f_idx not in final_all_frames_results:
                            final_all_frames_results[f_idx] = []
                        final_all_frames_results[f_idx].extend(valid_objs)

            id_offset += 1000

        finally:
            if session_id:
                try:
                    predictor.handle_request(request=dict(
                        type="close_session", session_id=session_id))
                except:
                    pass
            torch.cuda.empty_cache()

    predictor.shutdown()
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

    output_information = {
        "video_path": video_path,
        "inference_results": final_all_frames_results,
        "found_classes": target_nouns,
        "output_prefix": final_prefix,
        "mode": args.mode
    }

    print("--- Tracking finished.")

    return output_information
