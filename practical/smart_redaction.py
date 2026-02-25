import utils
from tqdm import tqdm
import gc
import traceback
import numpy as np
import models
import torch
import cv2
import os
from PIL import Image
from config import configs


# Memory Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Smart Redaction / Privacy
SMART_REDACTION_OBJECTS = [
    "face",
    "license plate",
    "credit card",
    "laptop screen",
    "mobile phone screen",
    "signature",
    "document",
    "id card"
]

try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
except ImportError:
    print("Warning: Could not import SAM 3 box utilities. Geometric prompting might fail.")

# LIMITS:
# If a SINGLE class has more than this many objects on Frame 0, we treat it with caution.
MAX_OBJECTS_PER_CLASS = 40
RESIZE_DIM = 480


def smart_redaction(image_path, args, prompt_text="face", blur_strength=51):
    """
    Privacy Tool: Detects objects matching 'prompt_text' and blurs them.
    Returns output_information dict for visualization.
    """
    print(f"\n--- Starting Smart Redaction on {image_path} ---")
    print(f"   > Target: '{prompt_text}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Image
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_data = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        raw_image = Image.open(image_path).convert('RGB')
        height, width = image_rgb.shape[:2]
    except Exception as e:
        print(f"Error loading image: {e}")
        return {}

    # Load SAM3
    print(">> Step 1: Loading SAM 3 for detection...")
    sam3_model, sam3_processor = models.load_sam3_model(
        device=device, conf_threshold=configs.SAM3_CONF_IMAGE_THRESHOLD)

    found_masks = []

    try:
        inference_state = sam3_processor.set_image(raw_image)

        normalized_text = prompt_text.replace(",", ".")
        prompts = [p.strip() for p in normalized_text.split('.') if p.strip()]
        len_prompts = len(prompts)

        for p in prompts:
            print(f"   > Scanning for '{p}'...")
            sam3_processor.reset_all_prompts(inference_state)
            inference_state = sam3_processor.set_text_prompt(
                state=inference_state, prompt=p
            )

            pred_masks = inference_state.get('masks')
            pred_scores = inference_state.get('scores')

            if pred_masks is None:
                continue

            if isinstance(pred_masks, torch.Tensor):
                pred_masks = pred_masks.reshape(-1, height, width)
                pred_scores = pred_scores.reshape(-1)

                keep_idxs = pred_scores > configs.SAM3_CONF_IMAGE_THRESHOLD
                pred_masks = pred_masks[keep_idxs]
                masks_np = pred_masks.cpu().numpy() > 0
            else:
                masks_np = pred_masks > 0

            for m in masks_np:
                found_masks.append({'segmentation': m})

        if not found_masks:
            print("   > No objects found to redact.")
            # Even if nothing found, return original image so pipeline continues
            return {
                "redacted_image_rgb": image_rgb,
                "output_prefix": os.path.join(args.output, os.path.splitext(os.path.basename(image_path))[0]),
                "mode": args.mode,
                "found": False
            }

        print(f"   > Found {len(found_masks)} areas to redact.")

        # Apply Redaction
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        redacted_image_rgb = apply_redaction(
            image_rgb, found_masks, kernel_size=(k_size, k_size))

    finally:
        del sam3_model
        del sam3_processor
        torch.cuda.empty_cache()

    # Pack Output Information
    base_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    output_prefix = os.path.join(args.output, name_without_ext)

    output_information = {
        "num_prompts": len_prompts,
        "redacted_image_rgb": redacted_image_rgb,
        "output_prefix": output_prefix,
        "mode": args.mode,
        "found": True
    }

    return output_information


def apply_redaction(image, masks_list, kernel_size=(51, 51)):
    """
    Applies a strong Gaussian Blur to the specific regions defined by the masks.
    Used for privacy/smart redaction.
    """
    if not masks_list:
        return image

    output_image = image.copy()

    # Combine all masks into one boolean map
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for m in masks_list:
        seg = m['segmentation']
        if seg.dtype != bool:
            seg = seg > 0
        combined_mask = np.logical_or(combined_mask, seg)

    if not np.any(combined_mask):
        return output_image

    # Blur the entire image heavily
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    output_image[combined_mask] = blurred_image[combined_mask]

    return output_image


def smart_redaction_video(video_path, args, prompt_text=None, blur_strength=51):
    """
    Video Privacy Tool: 
    1. Uses a predefined list of privacy objects (or user input).
    2. Propagates each object class independently through the video.
    3. Aggregates all masks.
    4. Renders a single redacted video.
    """
    print(f"\n--- Starting Smart Redaction VIDEO on {video_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(video_path):
        print("Error: Video path not found.")
        return {}

    # Determine Targets
    if prompt_text:
        final_prompts = [p.strip()
                         for p in prompt_text.split(',') if p.strip()]
    else:
        final_prompts = SMART_REDACTION_OBJECTS

    print(f"   > Targets: {final_prompts}")

    # Initialize SAM3 Video
    print(">> Step 1: Initializing SAM 3 Video Predictor...")
    predictor = models.load_sam3_video_model(device)

    # Get Video Info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cumulative_frame_masks = {}

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        # Loop through each privacy object class independently
        for noun in final_prompts:
            print(f"\n   > Processing Class: '{noun}'")

            response = predictor.handle_request(request=dict(
                type="start_session", resource_path=video_path
            ))
            session_id = response["session_id"]

            try:
                # Add prompt on Frame 0
                with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                    resp = predictor.handle_request(request=dict(
                        type="add_prompt", session_id=session_id, frame_index=0, text=noun
                    ))

                initial_objs = utils.parse_sam3_result(resp["outputs"], "0")

                if not initial_objs:
                    print(
                        f"     -> No '{noun}' detected on Frame 0. Skipping propagation.")
                    continue

                print(
                    f"     -> Found {len(initial_objs)} instances. Propagating...")

                # Propagate
                with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                    stream = predictor.handle_stream_request(request=dict(
                        type="propagate_in_video", session_id=session_id
                    ))

                    for r in tqdm(stream, total=total_frames, leave=False, desc=f"Tracking {noun}"):
                        f_idx = r["frame_index"]
                        out_data = r["outputs"]

                        masks_val = out_data.get("out_binary_masks")
                        scores_val = out_data.get("out_obj_scores")

                        if masks_val is not None:
                            # Apply Confidence Threshold safely for both Tensor and Numpy
                            if scores_val is not None:
                                if isinstance(scores_val, torch.Tensor):
                                    scores_cpu = scores_val.flatten().cpu()
                                    keep_indices = scores_cpu > configs.SAM3_CONF_SMART_REDACTION_VIDEO_THRESHOLD

                                    if isinstance(masks_val, np.ndarray):
                                        masks_val = masks_val[keep_indices.numpy(
                                        )]
                                    else:
                                        # Assuming masks_val is tensor
                                        masks_val = masks_val[keep_indices.to(
                                            masks_val.device)]

                            # Check for empty (Handle both Numpy and Tensor)
                            is_empty = False
                            if isinstance(masks_val, np.ndarray):
                                if masks_val.size == 0:
                                    is_empty = True
                            else:
                                if masks_val.numel() == 0:
                                    is_empty = True

                            if is_empty:
                                continue

                            # Convert to Numpy for processing
                            if isinstance(masks_val, torch.Tensor):
                                masks_np = (
                                    masks_val > 0.0).float().cpu().numpy()
                            else:
                                masks_np = masks_val

                            for i in range(masks_np.shape[0]):
                                m = masks_np[i]
                                if m.ndim == 3:
                                    m = m[0]
                                if m.sum() > 0:
                                    if f_idx not in cumulative_frame_masks:
                                        cumulative_frame_masks[f_idx] = []
                                    # Explicit 'segmentation' key for apply_redaction
                                    cumulative_frame_masks[f_idx].append(
                                        {'segmentation': m.astype(bool)})

            finally:
                if session_id:
                    predictor.handle_request(request=dict(
                        type="close_session", session_id=session_id))
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in SAM3 Propagation: {e}")
        traceback.print_exc()
    finally:
        predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Render Final Aggregated Video
    if not cumulative_frame_masks:
        print("   > No objects found to redact in the entire video.")
        return {
            "video_output_path": None,
            "original_video_path": video_path,
            "found": False
        }

    print(f"\n>> Step 2: Aggregating masks and Rendering...")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if prompt_text:
        output_filename = f"{base_name}_{prompt_text}_redacted.mp4"
    else:
        output_filename = f"{base_name}_redacted.mp4"
    output_path = os.path.join(args.output, output_filename)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in cumulative_frame_masks:
            frame_masks = cumulative_frame_masks[frame_idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            redacted_rgb = apply_redaction(
                frame_rgb, frame_masks, kernel_size=(k_size, k_size))
            final_frame = cv2.cvtColor(redacted_rgb, cv2.COLOR_RGB2BGR)
        else:
            final_frame = frame

        out.write(final_frame)
        frame_idx += 1

    cap.release()
    out.release()

    output_information = {
        "video_output_path": output_path,
        "original_video_path": video_path,
        "mode": args.mode,
        "found": True
    }

    print(f"successfully saved video on {output_path}")

    return output_information
