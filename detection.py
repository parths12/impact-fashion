from transformers import pipeline
from PIL import Image

# Path to config and checkpoint
CONFIG_FILE = 'cascade_rcnn_r50_fpn_1x_deepfashion2.py'
CHECKPOINT_FILE = 'cascade_rcnn_r50_fpn_1x_deepfashion2.pth'

# Define your clothing categories
CLOTHING_CATEGORIES = [
    "t-shirt", "shirt", "dress", "jeans", "jacket", "skirt", "shorts", "sweater", "coat", "blouse", "polo", "trousers", "suit", "hoodie"
]

# Load the zero-shot image classification pipeline
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

def detect_and_crop(image: Image.Image):
    results = model(image)
    crops = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
    return crops 

# New function to detect main clothing type
def detect_clothing_type(image: Image.Image):
    results = classifier(image, candidate_labels=CLOTHING_CATEGORIES)
    if results and 'labels' in results and len(results['labels']) > 0:
        return results['labels'][0]  # Most likely clothing type
    return None 