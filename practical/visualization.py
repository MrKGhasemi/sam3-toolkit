import os
import random
import matplotlib.pyplot as plt
import colorsys
from config import configs
import cv2
import numpy as np


def generate_harmonious_colors(n_classes):
    """
    Generates n distinct, bright, and harmonious colors using HSV space.
    Returns a list of (R, G, B) tuples.
    Updated to return RANDOMIZED colors on each call.
    """
    colors = []
    # Add a random start offset for hue to ensure uniqueness each run
    hue_start = random.random()

    for i in range(n_classes):
        # Offset the hue by the random start
        hue = (hue_start + (i / n_classes)) % 1.0
        saturation = 0.9 + (i % 2) * 0.1
        value = 0.9 + (i % 2) * 0.1

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    # Explicitly shuffle the list so colors aren't strictly sequential by spectrum
    random.shuffle(colors)
    return colors


def show_save_visualizations(output_information, figsize=(45, 30)):
    """
    Merges visualization and saving. 
    1. Randomizes colors for the display/save.
    2. Plots the figures in Notebook.
    3. Saves the exact same visualization maps to disk.
    """
    classified_masks = output_information.get("masks", [])
    image_rgb = output_information.get("image_rgb")
    class_colors = output_information.get("class_colors", [])
    output_prefix = output_information.get("output_prefix")
    mode = output_information.get("mode")

    if not classified_masks or image_rgb is None:
        print("No masks or image data found to visualize.")
        return

    # Randomization for Display & Save
    display_colors = list(class_colors)
    if len(display_colors) > 1:
        bg_color = display_colors[0]
        object_colors = display_colors[1:]
        random.shuffle(object_colors)
        display_colors = [bg_color] + object_colors

    # Generate Maps using the randomized colors
    instance_map, semantic_map = _create_maps(
        image_rgb, classified_masks, display_colors)
    legend_map = create_legend_map(image_rgb, classified_masks, display_colors)

    # DISPLAY
    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Raw Image", fontsize=30)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(instance_map)
    plt.title("Instance Labels (In-Place)", fontsize=30)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(semantic_map)
    plt.title("Semantic Map", fontsize=30)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(legend_map)
    plt.title("Segmentation with Legend", fontsize=30)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # SAVE
    if output_prefix:
        instance_map_bgr = cv2.cvtColor(instance_map, cv2.COLOR_RGB2BGR)
        semantic_map_bgr = cv2.cvtColor(semantic_map, cv2.COLOR_RGB2BGR)
        legend_map_bgr = cv2.cvtColor(legend_map, cv2.COLOR_RGB2BGR)

        mode_str = f"_{mode}" if mode else ""

        cv2.imwrite(f"{output_prefix}{mode_str}_instance.jpg",
                    instance_map_bgr)
        cv2.imwrite(f"{output_prefix}{mode_str}_semantic.png",
                    semantic_map_bgr)
        cv2.imwrite(f"{output_prefix}{mode_str}_legend.jpg",
                    legend_map_bgr)
        print(f"Saved visualizations to {output_prefix}{mode_str}_*.jpg/png")


def show_save_smart_reduction_visualizations(output_information, figsize=(20, 12)):
    """
    Displays and Saves the Smart Reduction result.
    """
    redacted_image_rgb = output_information.get("redacted_image_rgb")
    output_prefix = output_information.get("output_prefix")
    mode = output_information.get("mode")
    found = output_information.get("found", False)
    num_prompts = output_information.get("num_prompts")

    if redacted_image_rgb is None:
        print("No redacted image to visualize.")
        return

    # DISPLAY
    plt.figure(figsize=figsize)
    plt.imshow(redacted_image_rgb)
    status = "Redacted" if found else "Original (No Target Found)"
    plt.title(f"Smart Reduction: {status}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # SAVE
    if output_prefix:
        # Append unique timestamp to avoid overwrites
        output_filename = f"{output_prefix}_{mode}_redacted_{num_prompts}.jpg"
        cv2.imwrite(output_filename, cv2.cvtColor(
            redacted_image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved redacted image to: {output_filename}")


def create_legend_map(image, classified_masks, class_colors):
    """
    Creates a clean segmentation map with a legend.
    Updated to support multi-column layout for many classes.
    """
    output_image = image.copy()
    overlay = output_image.copy()

    # Draw Masks
    # Sort by area to ensure small objects are visible
    for mask_data in sorted(classified_masks, key=lambda x: x.get('area', 0), reverse=True):
        class_id = mask_data.get('class_id', 0)
        # Safety check for color index
        color = class_colors[class_id % len(class_colors)]
        mask = mask_data['segmentation'].astype(np.uint8)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay[mask.astype(bool)] = color

    alpha = configs.MASK_ALPHA
    output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)

    # Calculate Counts
    class_counts = {}
    class_id_map = {}
    for m in classified_masks:
        name = m['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1
        class_id_map[name] = m['class_id']

    if not class_counts:
        return output_image

    # Draw Legend (Multi-column)
    sorted_counts = sorted(class_counts.items(),
                           key=lambda x: x[1], reverse=True)

    img_h, img_w = image.shape[:2]
    base_scale = max(0.5, img_h / 1500.0)
    if len(sorted_counts) > 30:
        base_scale *= 0.8

    font_scale = base_scale
    font_thickness = max(1, int(font_scale * 2))
    line_spacing = int(35 * font_scale) + 5

    pad = 20
    available_h = img_h - (2 * pad)

    # Calculate dimensions
    legend_entries = []
    max_text_w = 0

    for name, count in sorted_counts:
        text = f"{name}: {count}"
        size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        max_text_w = max(max_text_w, size[0])
        legend_entries.append(
            (text, class_colors[class_id_map[name] % len(class_colors)]))

    col_w = max_text_w + int(60 * font_scale)  # width per column
    max_items_per_col = max(1, available_h // line_spacing)

    num_cols = (len(legend_entries) +
                max_items_per_col - 1) // max_items_per_col
    total_box_w = num_cols * col_w

    # Height is either full height (if multiple cols) or just enough for items
    if num_cols > 1:
        total_box_h = max_items_per_col * line_spacing + int(10 * font_scale)
    else:
        total_box_h = len(legend_entries) * \
            line_spacing + int(10 * font_scale)

    # Ensure background fits in image
    draw_w = min(total_box_w, img_w - 2*pad)
    draw_h = min(total_box_h, img_h - 2*pad)

    # Draw Background
    sub = output_image[pad: pad + draw_h, pad: pad + draw_w]
    if sub.shape[0] > 0 and sub.shape[1] > 0:
        white = np.full_like(sub, 255)
        output_image[pad: pad + draw_h, pad: pad +
                     draw_w] = cv2.addWeighted(sub, 0.3, white, 0.7, 0)

    # Draw Entries
    start_x = pad
    start_y = pad + line_spacing

    box_size = int(20 * font_scale)

    for i, (text, color) in enumerate(legend_entries):
        col_index = i // max_items_per_col
        row_index = i % max_items_per_col

        x = start_x + (col_index * col_w)
        y = start_y + (row_index * line_spacing)

        # Stop drawing if run out of screen width/height
        if x + col_w > img_w - pad:
            continue
        if y > img_h - pad:
            continue

        # Color Box
        cv2.rectangle(output_image,
                      (x + 10, y - box_size),
                      (x + 10 + box_size, y),
                      color, -1)

        # Text
        cv2.putText(output_image, text,
                    (x + 20 + box_size, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return output_image


def get_image_crops(raw_image):
    """
    Splits the image into 25 crops (16 patch, 4 quadrants, 2 Horizontal halves, 2 Vertical halves, 1 full)
    """
    x, y = raw_image.size
    p1 = raw_image.crop(box=(0, 0, x / 2, y / 2))
    p2 = raw_image.crop(box=(x / 2, 0, x, y / 2))
    p3 = raw_image.crop(box=(0, y / 2, x / 2, y))
    p4 = raw_image.crop(box=(x / 2, y / 2, x, y))
    p5 = raw_image.crop(box=(0, 0, x / 2, y))
    p6 = raw_image.crop(box=(x / 2, 0, x, y))
    p7 = raw_image.crop(box=(0, 0, x, y / 2))
    p8 = raw_image.crop(box=(0, y / 2, x, y))
    images = [p1, p2, p3, p4, p5, p6, p7, p8, raw_image]
    x_step = x / 4
    y_step = y / 4
    for i in range(4):
        for j in range(4):
            x1 = j * x_step
            y1 = i * y_step
            x2 = (j + 1) * x_step
            y2 = (i + 1) * y_step
            patch = raw_image.crop(box=(x1, y1, x2, y2))
            images.append(patch)

    return images


def _create_maps(image, classified_masks, class_colors):
    """Internal helper to generate the visualization images."""
    output_image_with_labels = image.copy()
    color_overlay = output_image_with_labels.copy()
    semantic_map = np.zeros(image.shape[:2], dtype=np.uint16)

    image_height = image.shape[0]
    font_scale = max(0.4, image_height / 1200.0)
    font_thickness = max(1, int(font_scale * 2))

    # Draw masks
    for mask_data in sorted(classified_masks, key=lambda x: x.get('area', 0), reverse=True):
        class_id = mask_data.get('class_id', 0)
        color = class_colors[class_id]

        mask = mask_data['segmentation'].astype(np.uint8)
        mask_resized = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        color_overlay[mask_resized] = color
        semantic_map[mask_resized] = class_id

    # Blend instance map
    alpha = configs.MASK_ALPHA
    output_image_with_labels = cv2.addWeighted(
        color_overlay, alpha, output_image_with_labels, 1 - alpha, 0)

    # Draw labels
    for mask_data in classified_masks:
        class_id = mask_data['class_id']
        class_name = mask_data['class_name']

        mask = mask_data['segmentation'].astype(np.uint8)
        mask_resized_for_centroid = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        M = cv2.moments(mask_resized_for_centroid)
        if M["m00"] == 0:
            bbox = mask_data['bbox']
            center_x = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
            center_y = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

        text_size, _ = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size

        text_x = center_x - text_w // 2
        text_y = center_y + text_h // 2

        padding = int(font_scale * 5)

        rect_x1 = text_x - padding
        rect_y1 = text_y - text_h - padding
        rect_x2 = text_x + text_w + padding
        rect_y2 = text_y + padding

        cv2.rectangle(output_image_with_labels, (rect_x1, rect_y1),
                      (rect_x2, rect_y2), class_colors[class_id], -1)
        cv2.putText(output_image_with_labels, class_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    semantic_color_map = np.zeros_like(image)
    for i, color in enumerate(class_colors):
        semantic_color_map[semantic_map == i] = color

    return output_image_with_labels, semantic_color_map


def semantic_edges_visualizations(output_information, figsize=(25, 15)):
    """
    Unified Semantic Edges Visualization.
    Handles:
    1. Image -> Show/Save BW and Color Maps.
    2. Video -> Renders specific Edge videos (BW & Color).
    """
    # VIDEO
    if "video_path" in output_information and output_information["video_path"]:
        print(f">> Rendering Video Semantic Edges...")
        render_video_semantic_edges(output_information)
        return

    # IMAGE
    edge_bw = output_information.get("edge_bw")
    edge_color_bgr = output_information.get("edge_color_bgr")
    output_prefix = output_information.get("output_prefix")
    mode = output_information.get("mode")

    if edge_bw is None or edge_color_bgr is None:
        print("No edge maps available to visualize.")
        return

    # Convert BGR to RGB for Matplotlib
    edge_color_rgb = cv2.cvtColor(edge_color_bgr, cv2.COLOR_BGR2RGB)

    # DISPLAY
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(edge_bw, cmap='gray')
    plt.title("Semantic Edges (B&W)", fontsize=16)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(edge_color_rgb)
    plt.title("Semantic Edges (Colored)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # SAVE
    if output_prefix:
        mode_str = f"_{mode}" if mode else ""
        cv2.imwrite(f"{output_prefix}{mode_str}_edges_bw.png", edge_bw)
        cv2.imwrite(
            f"{output_prefix}{mode_str}_edges_color.png", edge_color_bgr)
        print(f"Saved edge maps to {output_prefix}{mode_str}_edges_*.png")


def render_video_semantic_edges(output_information):
    """
    Specific renderer for Semantic Edges in video.
    Calculates morphological gradients (edges) per frame instead of just filling masks.
    """
    video_path = output_information["video_path"]
    inference_results = output_information.get("inference_results", {})
    output_prefix = output_information.get("output_prefix")
    found_classes = output_information.get("found_classes", [])
    mode = output_information.get("mode", "")
    thickness = output_information.get("thickness", 2)

    # Generate Colors
    palette = generate_harmonious_colors(len(found_classes))
    class_color_map = {name: palette[i]
                       for i, name in enumerate(found_classes)}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    suffix = f"_{mode}" if mode else ""
    out_bw_path = f"{output_prefix}{suffix}_edges_bw.mp4"
    out_col_path = f"{output_prefix}{suffix}_edges_color.mp4"

    # Writers for B&W and Colored Edges
    out_bw = cv2.VideoWriter(out_bw_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (w, h), isColor=False)
    out_col = cv2.VideoWriter(out_col_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (w, h), isColor=True)

    frame_idx = 0
    kernel = np.ones((thickness, thickness), np.uint8)

    print(f"   > Rendering edges for {len(inference_results)} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize blank canvases (Black background)
        frame_edge_bw = np.zeros((h, w), dtype=np.uint8)
        frame_edge_col = np.zeros((h, w, 3), dtype=np.uint8)

        # Get Objects for this frame
        frame_objs = inference_results.get(frame_idx, [])

        for obj in frame_objs:
            mask = obj['mask'].astype(np.uint8) * 255
            class_name = obj.get('class_name', 'Obj')

            # Resize mask if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Morphological Gradient
            edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

            # Add to Accumulators
            frame_edge_bw = cv2.add(frame_edge_bw, edges)

            # Colorize
            if class_name in class_color_map:
                c = class_color_map[class_name]
                color_bgr = (c[2], c[1], c[0])
                # Only paint where edges exist
                frame_edge_col[edges > 0] = color_bgr

        out_bw.write(frame_edge_bw)
        out_col.write(frame_edge_col)
        frame_idx += 1

    cap.release()
    out_bw.release()
    out_col.release()
    print(
        f"   > Saved edge videos to:\n     - {out_bw_path}\n     - {out_col_path}")


def create_clean_mask_map(image, masks_list, class_colors):
    output_image = image.copy()
    overlay = output_image.copy()
    for mask_data in sorted(masks_list, key=lambda x: x.get('area', 0), reverse=True):
        class_id = mask_data.get('class_id', 0)
        color = class_colors[class_id % len(class_colors)]
        mask = mask_data['segmentation'].astype(np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[mask.astype(bool)] = color
    alpha = 0.5
    final = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)
    return final  # Return clean image


def draw_geometric_inputs(image, prompt_value):
    out = image.copy()
    h, w = out.shape[:2]
    pos_color = (0, 255, 0)
    neg_color = (255, 0, 0)
    thickness = max(2, int(h/300))
    font_scale = max(0.6, h/1000)

    def draw_items(items, color, label_prefix):
        if not items:
            return
        if isinstance(items, (list, tuple, np.ndarray)):
            if len(items) > 0 and isinstance(items[0], (int, float, np.number)):
                items = [items]

        for i, item in enumerate(items):
            flat = list(map(float, item))
            if len(flat) == 4:
                x1, y1, x2, y2 = map(int, flat)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
                label = f"{label_prefix} Box {i+1}"
                cv2.putText(out, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            elif len(flat) == 2:
                x, y = map(int, flat)
                cv2.circle(out, (x, y), max(5, int(h/150)), color, -1)
                cv2.putText(out, f"{label_prefix} Pt {i+1}", (x+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if isinstance(prompt_value, dict):
        draw_items(prompt_value.get('positive', []), pos_color, "Pos")
        draw_items(prompt_value.get('negative', []), neg_color, "Neg")
        if 'box' in prompt_value:
            draw_items(prompt_value['box'], pos_color, "Pos")
        if 'point' in prompt_value:
            draw_items(prompt_value['point'], pos_color, "Pos")
    elif isinstance(prompt_value, (list, tuple, np.ndarray)):
        if len(prompt_value) > 0 and isinstance(prompt_value[0], (list, tuple, np.ndarray)):
            draw_items(prompt_value, pos_color, "Pos")
        else:
            draw_items(prompt_value, pos_color, "Pos")

    return out


def render_video_results(video_path, inference_results, class_colors, output_prefix, mode):
    """
    Renders video where:
    1. Counting Video: Bounding boxes colored by Class.
    2. Segmentation Video: Masks colored by Class.
    """
    print(f">> Starting Render for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not output_prefix:
        output_prefix = os.path.splitext(video_path)[0] + "_debug"
    if mode == 'llm':
        seg_out_path = f"{output_prefix}_{mode}_segmentation.mp4"
        count_out_path = f"{output_prefix}_{mode}_counting.mp4"
    else:
        seg_out_path = f"{output_prefix}_segmentation.mp4"
        count_out_path = f"{output_prefix}_counting.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_seg = cv2.VideoWriter(seg_out_path, fourcc, fps, (width, height))
    out_count = cv2.VideoWriter(count_out_path, fourcc, fps, (width, height))

    frame_idx = 0
    font_scale = max(0.6, height / 1000.0)
    font_thickness = max(2, int(font_scale * 3))

    total_frames = len(inference_results.keys())
    print(f"   > Data available for {total_frames} frames.")

    # duplicate classes detection & suppression
    all_classes = set()
    for f_objs in inference_results.values():
        for o in f_objs:
            all_classes.add(o.get('class_name', 'Object'))
    all_classes = list(all_classes)

    suppressed_classes = set()

    unique_ids_per_class = {}
    active_class_names = set()

    for f_objs in inference_results.values():
        for o in f_objs:
            c_name = o.get('class_name', 'Object')
            if c_name in suppressed_classes:
                continue
            active_class_names.add(c_name)
            if c_name not in unique_ids_per_class:
                unique_ids_per_class[c_name] = set()
            unique_ids_per_class[c_name].add(o['obj_id'])

    final_class_counts = {k: len(v) for k, v in unique_ids_per_class.items()}

    # Create mapping: Class -> {Global_ID -> 1-based Index}
    id_to_count_map = {}
    for cls_name, ids in unique_ids_per_class.items():
        sorted_ids = sorted(list(ids))
        for idx, global_id in enumerate(sorted_ids):
            id_to_count_map[global_id] = idx + 1

    sorted_active_classes = sorted(list(active_class_names))

    if class_colors and isinstance(class_colors, dict):
        class_color_map = class_colors
    else:
        class_palette_list = generate_harmonious_colors(
            len(sorted_active_classes))
        class_color_map = {name: class_palette_list[i]
                           for i, name in enumerate(sorted_active_classes)}

    # RENDER LOOP
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_objects = inference_results.get(frame_idx, [])
        valid_frame_objects = [o for o in frame_objects if o.get(
            'class_name') not in suppressed_classes]

        # Segmentation Video
        seg_frame = frame.copy()
        overlay = seg_frame.copy()

        for obj in valid_frame_objects:
            mask = obj['mask']
            class_name = obj.get('class_name', 'Obj')

            if class_name in class_color_map:
                c = class_color_map[class_name]
                color_bgr = (c[2], c[1], c[0])  # RGB tuple to BGR
            else:
                color_bgr = (0, 255, 0)

            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(
                    np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            overlay[mask] = color_bgr

        alpha = configs.MASK_ALPHA
        seg_frame = cv2.addWeighted(overlay, alpha, seg_frame, 1 - alpha, 0)
        out_seg.write(seg_frame)

        # Counting Video (Uses Class Color + Correct Label)
        count_frame = frame.copy()

        for obj in valid_frame_objects:
            obj_id = obj['obj_id']
            class_name = obj.get('class_name', 'Obj')

            count_label = id_to_count_map.get(obj_id, "?")
            display_text = f"{count_label}"

            if class_name in class_color_map:
                c = class_color_map[class_name]
                color_bgr = (c[2], c[1], c[0])
            else:
                color_bgr = (0, 255, 0)

            mask = obj['mask']
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()

                # Box
                cv2.rectangle(count_frame, (x1, y1), (x2, y2), color_bgr, 2)

                text_size, _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_w, text_h = text_size
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > 10 else y1 + text_h + 5

                # Label BG
                cv2.rectangle(count_frame, (text_x, text_y - text_h - 2),
                              (text_x + text_w, text_y + 2), color_bgr, -1)
                # Label Text (White)
                cv2.putText(count_frame, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Draw Legend
        legend_x = 20
        legend_y = 40
        line_height = int(30 * (font_scale / 0.6))

        cv2.putText(count_frame, "Total Counts:", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness+1)
        legend_y += line_height

        for cls_name, count in final_class_counts.items():
            c = class_color_map.get(cls_name, (255, 255, 255))
            color_bgr = (c[2], c[1], c[0])

            cv2.rectangle(count_frame, (legend_x, legend_y - line_height + 5),
                          (legend_x + 20, legend_y - 5), color_bgr, -1)
            text = f" {cls_name}: {count}"
            cv2.putText(count_frame, text, (legend_x + 25, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness+2)
            cv2.putText(count_frame, text, (legend_x + 25, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            legend_y += line_height

        out_count.write(count_frame)
        frame_idx += 1

    cap.release()
    out_seg.release()
    out_count.release()
    print(
        f"   > Saved videos:\n     - {seg_out_path}\n     - {count_out_path}")


def counting_visualizations(output_information, figsize=(30, 15)):
    """
    Unified visualization function for Counting tasks.
    Layouts:
    1. Automatic/Prompt -> [Segmentation Map] | [Counting Map]
    2. Geometric        -> [Raw Image] | [Geometric Inputs] | [Counting Map]
    """
    # Task Context
    task_type = output_information.get("task_type", "automatic")
    is_video = "video_path" in output_information or "inference_results" in output_information
    mode = output_information.get("mode", "blip")
    output_prefix = output_information.get("output_prefix")

    # Color Generation
    found_classes = output_information.get("found_classes", [])
    if not found_classes and not is_video:
        masks = output_information.get("masks", [])
        found_classes = list(set([m['class_name'] for m in masks]))

    palette = generate_harmonious_colors(len(found_classes))
    class_color_map = {name: palette[i]
                       for i, name in enumerate(found_classes)}

    # IMAGE
    if not is_video:
        image_rgb = output_information.get("image_rgb")
        masks = output_information.get("masks", [])
        input_map = output_information.get("input_map")  # For geometric mode

        if image_rgb is None or not masks:
            print("No data to visualize for image counting.")
            return

        # Create Right Image: Counting Map (Boxes + Counts)
        counting_box_map = create_counting_map(
            image_rgb, masks, class_color_map)

        # Create Left Image: Segmentation Map (Masks + Legend)
        segmentation_map = create_segmentation_legend_map(
            image_rgb, masks, class_color_map)

        # Display Layout
        plt.figure(figsize=figsize)

        if task_type == "geometric" and input_map is not None:
            # Geometric Layout: [Raw] [Inputs] [Boxes] (Preserving 3-image layout)
            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            plt.title("Raw Image", fontsize=15)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(input_map)
            plt.title("Geometric Inputs", fontsize=15)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(counting_box_map)
            plt.title("Counting Result", fontsize=15)
            plt.axis('off')
        else:
            # Auto/Prompt Layout: [Segmentation] [Boxes] (Requested 2-image layout)
            plt.subplot(1, 2, 1)
            plt.imshow(segmentation_map)
            plt.title("Segmentation Result", fontsize=15)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(counting_box_map)
            plt.title("Counting Result", fontsize=15)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save
        if output_prefix:
            mode_str = f"_{mode}" if mode else ""

            # Save Boxes
            cv2.imwrite(f"{output_prefix}{mode_str}_counting_boxes.jpg", cv2.cvtColor(
                counting_box_map, cv2.COLOR_RGB2BGR))

            # Save Segmentation
            cv2.imwrite(f"{output_prefix}{mode_str}_counting_segmentation.jpg", cv2.cvtColor(
                segmentation_map, cv2.COLOR_RGB2BGR))

            # Save Geometric Input if exists
            if input_map is not None:
                cv2.imwrite(f"{output_prefix}{mode_str}_geometric_inputs.jpg", cv2.cvtColor(
                    input_map, cv2.COLOR_RGB2BGR))

            print(
                f"Saved counting visualizations to {output_prefix}{mode_str}_*.jpg")

    # VIDEO LOGIC
    else:
        video_path = output_information.get("video_path")
        inference_results = output_information.get("inference_results", {})
        if video_path and inference_results:
            render_video_results(video_path, inference_results,
                                 class_color_map, output_prefix, mode)


def create_segmentation_legend_map(image, masks, class_color_map):
    """
    Creates the Segmentation Map (Masks + Legend) using the dictionary color map.
    This ensures it matches the colors used in the Bounding Box map.
    """
    output_image = image.copy()
    overlay = output_image.copy()

    # 1. Draw Masks
    for mask_data in sorted(masks, key=lambda x: x.get('area', 0), reverse=True):
        class_name = mask_data.get('class_name')
        # Default green if missing
        color = class_color_map.get(class_name, (0, 255, 0))

        mask = mask_data['segmentation'].astype(np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay[mask.astype(bool)] = color

    alpha = configs.MASK_ALPHA
    output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)

    # Calculate counts to show in legend
    class_counts = {}
    for m in masks:
        name = m['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1

    legend_dict = {f"{k}: {v}": class_color_map.get(
        k, (255, 255, 255)) for k, v in class_counts.items()}

    # 3. Draw Legend
    return draw_opencv_legend(output_image, legend_dict)


def create_counting_map(image_rgb, masks, class_color_map):
    """
    Draws Bounding Boxes, Counts on boxes, and Legend.
    """
    output_image = image_rgb.copy()
    h, w = output_image.shape[:2]
    font_scale = max(0.5, h / 1000.0)
    thickness = max(2, int(font_scale * 3))

    class_counters = {}
    # Sort masks for consistent numbering
    sorted_masks = sorted(masks, key=lambda x: (x['bbox'][1], x['bbox'][0]))

    for m in sorted_masks:
        class_name = m.get('class_name', 'object')
        bbox = m['bbox']
        class_counters[class_name] = class_counters.get(class_name, 0) + 1
        obj_count_id = class_counters[class_name]

        color_rgb = class_color_map.get(class_name, (0, 255, 0))

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color_rgb, thickness)

        label_text = f"{obj_count_id}"
        (text_w, text_h), _ = cv2.getTextSize(label_text,
                                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.rectangle(output_image, (x1, y1 - text_h - 4),
                      (x1 + text_w + 4, y1), color_rgb, -1)
        cv2.putText(output_image, label_text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Add Legend
    legend_dict = {f"{k}: {v}": class_color_map.get(
        k, (255, 255, 255)) for k, v in class_counters.items()}
    return draw_opencv_legend(output_image, legend_dict)


def draw_opencv_legend(image, legend_dict):
    """
    Draws a compact legend on the image.
    """
    if not legend_dict:
        return image
    img_h, _ = image.shape[:2]
    font_scale = max(0.4, img_h / 2000.0)
    thickness = max(1, int(font_scale * 1.5))
    line_spacing = int(20 * (font_scale / 0.4)) + 5

    max_text_w = 0
    for text in legend_dict.keys():
        size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_text_w = max(max_text_w, size[0])

    box_w = max_text_w + 35
    box_h = len(legend_dict) * line_spacing + 10

    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (255, 255, 255), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    y = 5 + line_spacing - 5
    for text, color in legend_dict.items():
        c_int = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(image, (10, y - int(10*font_scale)), (20, y), c_int, -1)
        cv2.putText(image, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_spacing
    return image


def abds():
    pass
