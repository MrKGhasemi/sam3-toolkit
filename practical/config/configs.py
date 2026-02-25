SAM3_CHECKPOINT_PATH = "weights/sam3.pt"  # Path to SAM3 checkpoint
SAM3_VOCAB_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"  # Path to SAM3 BPE vocab
# -----------------------------------------------------

# Model Settings
SAM3_CONF_IMAGE_THRESHOLD = 0.3  # Confidence to accept a mask
SAM3_CONF_IMAGE_FOR_COUNTING = 0.2

# Confidence to accept a mask in video
SAM3_CONF_SMART_REDACTION_VIDEO_THRESHOLD = 0.2
SAM3_CONF_OBJECT_COUNTING_VIDEO_THRESHOLD = 0.2
SAM3_CONF_VIDEO_SEARCH_ENGINE_THRESHOLD = 0.2

SAM3_IMAGE_SEGMENTATION_NMS_THRESHOLD = 0.8
# -----------------------------------------------------

# NLP & Captioning
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
SPACY_MODEL = "en_core_web_lg"
BLIP_MAX_TOKENS = 50  # Max tokens for BLIP captioning
SYNONYM_THRESHOLD = 0.6  # Similarity threshold for synonym merging
# -----------------------------------------------------

# LLM-Models
LLM_CAPTION_MODEL_NAME = "gpt-5-chat"
LLM_PARSER_MODEL_NAME = "gpt-5.2-chat"

# LLM-Captioner-SYSTEM-PROMPT
LLM_CAPTION_SYSTEM_PROMPT = """You are an object detection assistant. 
generate a detailed caption for the provided image, describing all distinct, physical objects in the Image (e.g., person, car, ball, bicycle), even cloud, sky and ground types. 
Return ONLY the caption without any additional text."""
LLM_CAPTION_USER_PROMPT = "give the caption for this image including all objects"

# LLM-Noun-Parser-SYSTEM-PROMPT(Image Segmentation)
LLM_IMAGE_SEGMENTATION_PARSER_SYSTEM_PROMPT = """You are an expert at semantic data cleaning and object taxonomy. 
Your task is to take a list of raw noun phrases extracted from image captions and transform them into a clean, distinct list of object classes.
Guidelines:
1. Adjectives: Remove adjectives attributes like color, make, or type (e.g., "red car" into "car", "evergreen tree" into "tree", "green bicyle" into "bicycle", "green rental bicyle" into "bicyle").
2. Remove Redundancy & Noise: Clean out repetitive patterns like "car car car" into "car".
3. Prevent Massive Captions: Reduce the massive captions like "trees with green foliage", into "tree" and "foliage".
4. Specific Names: If the list contains specific name and brands like "black car" and also "black audi car", keep them all and devide the name/brand separate like "black car" and "audi".
5. Standardize Persons: Convert "man", "woman", "pedestrian", or "child" into "person" unless the specific distinction is visually critical for the scene.
6. Remove Body Parts: Eliminate body parts like "arm", "head", "eye", "mouth", etc. unless they are the primary focus.
7. Output Format: Return ONLY a valid list of strings. No explanations."""

# LLM-Noun-Parser-SYSTEM-PROMPT(Video Search Engine)
LLM_SEARCH_ENGINE_PARSER_SYSTEM_PROMPT = """You are an expert at semantic data cleaning and object taxonomy. 
Your task is to take a list of raw noun phrases extracted from image captions and transform them into a clean, distinct list of object classes.
Guidelines:
1. Preserve Descriptive Adjectives: Retain important visual attributes like color, make, or type (e.g., "red car", "evergreen tree", "audi sedan").
2. Remove Redundancy & Noise: Clean out repetitive patterns like "car car car" into "car".
3. Prevent Massive Captions: Reduce the massive captions like "trees with green foliage", keep "trees with green foliage" and add "tree" and "foliage" clases either.
4. Specific Names: If the list contains specific name and brands like "black car" and also "black audi car", keep them all and devide the name/brand separate like "black car" and "black audi car" and "audi".
5. Same Objects Semantically: If the list has "green bicyle" and "green rental bicyle" which are semanticlly point to same objects but not contains specific names or brands, 
                              keep all and add specific objects like "green bicyle" and "green rental bicyle" and "bicyle".
6. Standardize Persons: Convert "man", "woman", "pedestrian", or "child" into "person" unless the specific distinction is visually critical for the scene.
7. Output Format: Return ONLY a valid list of strings. No explanations."""
# -----------------------------------------------------

# Video Search Engine
VECTOR_DB_PATH = "notebook_outputs/vector_db"
INDEX_FILE = "video_search.index"
METADATA_FILE = "video_metadata.pkl"
# -----------------------------------------------------

# Output
SAVE_FIGURES = True
MASK_ALPHA = 0.5  # Transparency for mask overlays
