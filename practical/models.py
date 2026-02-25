import os
import sam3
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from config import configs
import torch


def load_nlp_models(device="cuda"):
    """
    Loads auxiliary NLP models (BLIP, SpaCy) if you need auto-text generation.
    """
    print("Loading NLP models...")

    blip_model = BlipForConditionalGeneration.from_pretrained(
        configs.BLIP_MODEL_ID).to(device)

    blip_processor = BlipProcessor.from_pretrained(configs.BLIP_MODEL_ID)

    nlp = spacy.load(configs.SPACY_MODEL)

    return blip_model, blip_processor, nlp


def load_sam3_model(conf_threshold, device="cuda"):
    """
    Loads the SAM 3 model and processor.
    """
    print(f"Loading SAM 3 model from {configs.SAM3_CHECKPOINT_PATH}...")

    sam3_root = os.path.dirname(sam3.__file__)
    bpe_full_path = os.path.join(sam3_root, "..", configs.SAM3_VOCAB_PATH)

    if not os.path.exists(bpe_full_path):
        bpe_full_path = configs.SAM3_VOCAB_PATH

    model = build_sam3_image_model(
        bpe_path=bpe_full_path,
        checkpoint_path=configs.SAM3_CHECKPOINT_PATH,
        load_from_HF=False
    )

    model.to(device)
    model.eval()

    processor = Sam3Processor(
        model, confidence_threshold=conf_threshold)

    print("SAM 3 Model & Processor loaded successfully.")
    return model, processor


def load_sam3_video_model(device="cuda"):
    print(f"Loading SAM 3 Video Predictor on {device}...")
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "..", configs.SAM3_VOCAB_PATH)
    if not os.path.exists(bpe_path):
        bpe_path = configs.SAM3_VOCAB_PATH

    # The MultiGPU wrapper lacks 'init_state' in some versions.
    if device == "cuda" and torch.cuda.device_count() > 1:
        # Use explicit GPUs list for multi-gpu
        gpus_to_use = [i for i in range(torch.cuda.device_count())]
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            bpe_path=bpe_path,
            checkpoint_path=configs.SAM3_CHECKPOINT_PATH
        )
    else:
        predictor = build_sam3_video_predictor(
            bpe_path=bpe_path,
            checkpoint_path=configs.SAM3_CHECKPOINT_PATH
        )

    return predictor
