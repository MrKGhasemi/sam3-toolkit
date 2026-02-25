import difflib
import open_clip
import os
import cv2
import torch
import numpy as np
import faiss
import pickle
import gc
import re
from PIL import Image
from tqdm import tqdm
import models
from config import configs
import utils
import class_generators
from matplotlib import pyplot as plt
import matplotlib.patches as patches

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global Cache
GLOBAL_CLIP_EMBEDDER = None


class CLIPEmbedder:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading OpenCLIP ViT-g/14 (1024 dims) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-g-14', pretrained='laion2b_s34b_b88k', device=self.device
        )
        self.model = self.model.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')
        self.model.eval()

    def get_image_embedding(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.float().cpu().numpy()

    def get_text_embedding(self, text):
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float().cpu().numpy()


def get_shared_clip_embedder():
    """Singleton pattern to avoid reloading CLIP."""
    global GLOBAL_CLIP_EMBEDDER
    if GLOBAL_CLIP_EMBEDDER is None:
        GLOBAL_CLIP_EMBEDDER = CLIPEmbedder(device)
    return GLOBAL_CLIP_EMBEDDER


def build_video_index(video_path, args):
    print(
        f"\n--- Starting Semantic Video Indexing for: {video_path} ---")

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Metadata
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ------------------------------------------------------------
    # --- Phase 1: SCANNING
    scan_interval = 15
    print(f">> Phase 1: Scanning video every {scan_interval} frames...")

    blip_model, blip_processor, nlp = models.load_nlp_models(device)
    detected_classes = {}
    cap = cv2.VideoCapture(video_path)

    try:
        for frame_idx in tqdm(range(0, total_frames, scan_interval), desc="Scanning"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            current_nouns = []
            try:
                if args.mode == "blip":
                    current_nouns, _ = class_generators.get_classes_blip(
                        pil_image, {
                            "blip_model": blip_model, "blip_processor": blip_processor, "nlp": nlp}, device
                    )
                elif args.mode == "llm":
                    temp_path = os.path.join(
                        args.output, f"temp_{frame_idx}.jpg")
                    cv2.imwrite(temp_path, frame)
                    current_nouns, _ = class_generators.get_classes_llm(
                        True, temp_path, {
                            "nlp": nlp}, args.api_key, args.base_url,
                        configs.LLM_SEARCH_ENGINE_PARSER_SYSTEM_PROMPT
                    )
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                print(f"Scanning error: {e}")
                continue

            if current_nouns:
                for noun in current_nouns:
                    if len(noun) > 1 and noun not in detected_classes:
                        detected_classes[noun] = frame_idx
    finally:
        cap.release()

    # Unload
    del blip_model, blip_processor, nlp
    gc.collect()
    torch.cuda.empty_cache()

    # Cleaning
    new_detected_classes = utils.search_engine_clean_nouns(
        list(detected_classes.keys()), args.api_key, args.base_url,
        configs.LLM_SEARCH_ENGINE_PARSER_SYSTEM_PROMPT)

    if not new_detected_classes:
        print("CRITICAL WARNING: No classes found.")
        return

    print(f"detected classes after cleaning: {new_detected_classes}")

    # ------------------------------------------------------------
    # --- Phase 2: TRACKING
    print(">> Phase 2: Tracking...")
    predictor = models.load_sam3_video_model(device=device)
    track_registry = {}

    final_tracking_list = []
    for clean_name in new_detected_classes:
        start_f = 0
        for raw_k, raw_f in detected_classes.items():
            if clean_name in raw_k or raw_k in clean_name:
                start_f = raw_f
                break
        final_tracking_list.append((clean_name, start_f))

    for class_name, start_frame in tqdm(final_tracking_list, desc="Tracking"):
        session_id = None
        try:
            response = predictor.handle_request(request=dict(
                type="start_session", resource_path=video_path))
            session_id = response["session_id"]

            predictor.handle_request(request=dict(
                type="add_prompt", session_id=session_id, frame_index=start_frame, text=class_name
            ))

            stream = predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=session_id))
            track_cap = cv2.VideoCapture(video_path)

            for result in stream:
                out_frame_idx = result["frame_index"]
                frame_objects = utils.parse_sam3_result(
                    result["outputs"], str(out_frame_idx))
                if not frame_objects:
                    continue

                frame_loaded = False
                current_frame_img = None

                for obj in frame_objects:
                    if obj['mask'].sum() < 200:
                        continue
                    if 'score' in obj and obj['score'] < configs.SAM3_CONF_VIDEO_SEARCH_ENGINE_THRESHOLD:
                        continue

                    unique_track_id = f"{class_name}_{obj['obj_id']}"
                    if unique_track_id not in track_registry:
                        track_registry[unique_track_id] = {
                            "class_name": class_name, "obj_id": obj['obj_id'],
                            "trajectory": [], "best_crop": None, "max_area": 0, "best_frame_idx": 0
                        }

                    y_idx, x_idx = np.where(obj['mask'])
                    x1, x2 = x_idx.min(), x_idx.max()
                    y1, y2 = y_idx.min(), y_idx.max()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    area = (x2 - x1) * (y2 - y1)

                    track_registry[unique_track_id]["trajectory"].append(
                        {"frame": out_frame_idx, "bbox": bbox})

                    if area > track_registry[unique_track_id]["max_area"]:
                        if not frame_loaded:
                            track_cap.set(
                                cv2.CAP_PROP_POS_FRAMES, out_frame_idx)
                            _, f = track_cap.read()
                            current_frame_img = cv2.cvtColor(
                                f, cv2.COLOR_BGR2RGB)
                            frame_loaded = True

                        pad = 10
                        width = int(track_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(track_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                        cx2, cy2 = min(width, x2+pad), min(height, y2+pad)
                        track_registry[unique_track_id]["best_crop"] = current_frame_img[cy1:cy2, cx1:cx2].copy(
                        )
                        track_registry[unique_track_id]["max_area"] = area
                        track_registry[unique_track_id]["best_frame_idx"] = out_frame_idx

            track_cap.release()
        except Exception as e:
            print(f"Error tracking {class_name}: {e}")
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

    # ------------------------------------------------------------
    # --- Phase 3: EMBEDDING
    print(
        f">> Phase 3: Generating Embeddings for {len(track_registry)} tracks...")
    clip_embedder = get_shared_clip_embedder()

    embeddings_list = []
    metadata_list = []

    for track_id, data in tqdm(track_registry.items(), desc="Embedding"):
        if data["best_crop"] is None:
            continue

        vector = clip_embedder.get_image_embedding(data["best_crop"])
        vector = vector.flatten()

        metadata_list.append({
            "track_id": track_id, "class_name": data["class_name"], "obj_id": data["obj_id"],
            "video_path": video_path, "trajectory": data["trajectory"],
            "best_crop_preview": data["best_crop"], "best_frame_idx": data.get("best_frame_idx", 0)
        })
        embeddings_list.append(vector)

    # ------------------------------------------------------------
    # --- Phase 4: SAVE
    if embeddings_list:
        print(f">> Phase 4: Saving Index ({len(embeddings_list)} tracks)...")
        emb_matrix = np.vstack(embeddings_list).astype('float32')
        index = faiss.IndexFlatL2(emb_matrix.shape[1])
        index.add(emb_matrix)

        if not os.path.exists(configs.VECTOR_DB_PATH):
            os.makedirs(configs.VECTOR_DB_PATH)
        faiss.write_index(index, os.path.join(
            configs.VECTOR_DB_PATH, configs.INDEX_FILE))
        with open(os.path.join(configs.VECTOR_DB_PATH, configs.METADATA_FILE), 'wb') as f:
            pickle.dump(metadata_list, f)
        print("--- Indexing Complete! ---")
    else:
        print("No tracks found.")


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


def is_duplicate_track(meta1, meta2, iou_threshold=0.7, time_overlap_threshold=0.5):
    """
    Checks if two tracks represent the same object.
    Criteria:
    1. Significant overlap in time (frame range).
    2. High Spatial IoU on the overlapping frames.
    """
    # Fast check: Exact trajectory match
    if meta1['trajectory'] == meta2['trajectory']:
        return True

    # Map frame  to bbox for fast lookup
    t1_map = {t['frame']: t['bbox'] for t in meta1['trajectory']}
    t2_map = {t['frame']: t['bbox'] for t in meta2['trajectory']}

    frames1 = set(t1_map.keys())
    frames2 = set(t2_map.keys())

    # Check temporal intersection
    shared_frames = frames1.intersection(frames2)
    if not shared_frames:
        return False

    # Check temporal overlap ratio (relative to the shorter track)
    min_len = min(len(frames1), len(frames2))
    if len(shared_frames) / min_len < time_overlap_threshold:
        return False

    # Check Spatial IoU on a sample of shared frames to save time
    sorted_shared = sorted(list(shared_frames))
    step = max(1, len(sorted_shared) // 10)
    sample_frames = sorted_shared[::step]

    ious = []
    for f in sample_frames:
        ious.append(utils.calculate_box_iou_internal(t1_map[f], t2_map[f]))

    avg_iou = sum(ious) / len(ious)
    return avg_iou > iou_threshold


def render_track_video(metadata_entry, output_folder="outputs", overlay_text=None):
    """
    Renders a video clip with Drift Detection.
    If the box jumps too fast (indicative of ID Switch), it stops drawing the box.
    """
    video_path = metadata_entry['video_path']
    trajectory = metadata_entry['trajectory']
    track_id = metadata_entry['track_id']

    label = overlay_text if overlay_text else track_id

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    trajectory.sort(key=lambda x: x['frame'])
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{label}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_map = {item['frame']: item['bbox'] for item in trajectory}
    start_frame = trajectory[0]['frame']
    end_frame = trajectory[-1]['frame']

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_center = None
    drift_threshold = width * 0.1  # 10% screen width jump per frame allowed

    for f_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx in frame_map:
            bbox = frame_map[f_idx]

            # drift check
            current_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            is_drifting = False

            if prev_center is not None:
                dist = np.sqrt(
                    (current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
                if dist > drift_threshold:
                    is_drifting = True

            prev_center = current_center

            # Draw Box
            if not is_drifting:
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (0, 255, 0), 4)
                text_y = max(30, bbox[1]-10)
                cv2.putText(
                    frame, label, (bbox[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(
                    frame, "Lost/Drifted", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        out.write(frame)
    cap.release()
    out.release()
    return out_path


def search_video(query_text, top_k=2, label_threshold=0.3, save_video=True):
    print(
        f"\n--- Searching for: '{query_text}' (Threshold: {label_threshold}) ---")

    clip_embedder = get_shared_clip_embedder()
    query_vec = clip_embedder.get_text_embedding(
        query_text).flatten().astype('float32').reshape(1, -1)

    index_path = os.path.join(configs.VECTOR_DB_PATH, configs.INDEX_FILE)
    meta_path = os.path.join(configs.VECTOR_DB_PATH, configs.METADATA_FILE)

    if not os.path.exists(index_path):
        print("Index missing.")
        return

    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    # Candidate Retrieval
    search_pool_size = max(50, top_k * 5)
    distances, indices = index.search(query_vec, search_pool_size)

    candidates = []

    query_terms = set(query_text.lower().split())
    query_terms = {t for t in query_terms if len(t) > 2}

    # Score & Fuzzy Boost
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        meta = metadata[idx]
        l2_dist = distances[0][i]
        final_score = 1 - (l2_dist / 2)

        class_name_lower = meta['class_name'].lower()
        class_words = class_name_lower.split()
        boost_applied = False

        for q_term in query_terms:
            if q_term in class_name_lower:
                final_score += 0.20
                boost_applied = True
                break

            for c_word in class_words:
                similarity = difflib.SequenceMatcher(
                    None, q_term, c_word).ratio()
                if similarity > 0.80:
                    final_score += 0.20
                    boost_applied = True
                    break
            if boost_applied:
                break

        if final_score < label_threshold:
            continue

        candidates.append({
            "meta": meta,
            "score": final_score,
            "boosted": boost_applied
        })

    # Sort & NMS
    candidates.sort(key=lambda x: x['score'], reverse=True)

    final_results = []
    for cand in candidates:
        is_duplicate = False
        for accepted in final_results:
            if is_duplicate_track(cand['meta'], accepted['meta']):
                is_duplicate = True
                break
        if not is_duplicate:
            final_results.append(cand)
        if len(final_results) >= top_k:
            break

    if not final_results:
        print("No matches found above threshold.")
        return

    for i, res in enumerate(final_results):
        meta = res['meta']
        cap = cv2.VideoCapture(meta['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        best_frame = meta.get('best_frame_idx', 0)
        timestamp = best_frame / fps
        timestamp_str = f"{int(timestamp // 60):02}:{int(timestamp % 60):02}"

        # Capture Frame for Visualization
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, frame = cap.read()
        cap.release()

        print(
            f"\n[RESULT {i+1}] Score: {res['score']:.4f} {'(Boosted)' if res['boosted'] else ''} | ID: {meta['track_id']}")
        print(f"  - Class: {meta['class_name']} | Time: {timestamp_str}")

        # visualization
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(img_rgb)

            best_bbox = next(
                (t['bbox'] for t in meta['trajectory'] if t['frame'] == best_frame), None)

            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=3, edgecolor='#00ff00', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, max(0, y1-10), f"{query_text}", color='white', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='#00ff00', edgecolor='none', alpha=0.7))

            ax.set_title(
                f"Result {i+1}: {query_text} @ {timestamp_str} (Score: {res['score']:.2f})")
            ax.axis('off')
            plt.show()

        duration = len(meta['trajectory']) / fps
        if save_video and duration >= 1.5:
            vid_path = render_track_video(
                meta, output_folder="notebook_outputs/vector_db/outputs", overlay_text=query_text)
            print(f"  > Saved: {vid_path} ({duration:.2f}s)")
        else:
            reason = "User Disabled" if not save_video else "Too Short (<1.5s)"
            print(f"  > Video Skipped: {reason} (Duration: {duration:.2f}s)")
