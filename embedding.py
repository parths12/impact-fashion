from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def get_clip_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()

def get_clip_text_embedding(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().flatten() 