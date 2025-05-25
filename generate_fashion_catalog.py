import os
import pandas as pd
from PIL import Image
from embedding import get_clip_embedding
import logging
import csv
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parser for dataset selection
parser = argparse.ArgumentParser(description='Generate fashion catalog for selected dataset.')
parser.add_argument('--dataset', choices=['main', 'myntra'], default='main', help='Dataset to use: main or myntra (default: main)')
parser.add_argument('--max_images', type=int, default=1000, help='Maximum number of images to process (default: 1000)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
args = parser.parse_args()

if args.dataset == 'main':
    DATASET_DIR = 'dataset/images'
    STYLES_CSV = 'dataset/styles.csv'
elif args.dataset == 'myntra':
    DATASET_DIR = 'dataset/myntradataset/images'
    STYLES_CSV = 'dataset/myntradataset/styles.csv'
else:
    raise ValueError('Invalid dataset selection!')

CATALOG_CSV = 'catalog.csv'

def resize_image(image, max_size=224):
    """Resize image while maintaining aspect ratio"""
    ratio = min(max_size/image.size[0], max_size/image.size[1])
    new_size = tuple(int(dim * ratio) for dim in image.size)
    return image.resize(new_size, Image.Resampling.LANCZOS)

def process_batch(batch_rows):
    """Process a batch of images and return their embeddings"""
    batch_embeddings = []
    for row in batch_rows:
        img_path = os.path.join(DATASET_DIR, row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            img = resize_image(img)  # Resize image for faster processing
            emb = get_clip_embedding(img)
            emb_str = ','.join(map(str, emb))
            logger.debug(f"Successfully processed {img_path}")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            emb_str = ','.join(['0']*512)
        batch_embeddings.append(emb_str)
    return batch_embeddings

# Read the styles CSV with error handling
logger.info(f"Reading styles from {STYLES_CSV}")
try:
    df = pd.read_csv(STYLES_CSV, 
                     dtype={'id': str},
                     on_bad_lines='skip',
                     encoding='utf-8')
    
    # Limit the number of images for faster processing
    if len(df) > args.max_images:
        df = df.sample(n=args.max_images, random_state=42)
        logger.info(f"Limited dataset to {args.max_images} images")
    
    logger.info(f"Processing {len(df)} entries from styles.csv")
except Exception as e:
    logger.error(f"Error reading CSV: {e}")
    raise

def find_image_file(image_id):
    image_id = str(image_id).split('.')[0]
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        path = os.path.join(DATASET_DIR, f"{image_id}.{ext}")
        if os.path.exists(path):
            return f"{image_id}.{ext}"
    return None

# Find image paths
logger.info("Finding image files...")
df['image_path'] = df['id'].apply(find_image_file)
df = df[df['image_path'].notnull()]

logger.info(f"Found {df['image_path'].notnull().sum()} images")

# Process images in batches
logger.info(f"Generating embeddings in batches of {args.batch_size}...")
all_embeddings = []
for i in tqdm(range(0, len(df), args.batch_size)):
    batch = df.iloc[i:i + args.batch_size]
    batch_embeddings = process_batch([row for _, row in batch.iterrows()])
    all_embeddings.extend(batch_embeddings)

# Prepare final dataframe
logger.info("Preparing final catalog...")
df['name'] = df['productDisplayName']
df['price'] = 999
df['embedding'] = all_embeddings

# Save catalog
output_df = df[['id', 'name', 'price', 'image_path', 'embedding', 'gender', 'articleType', 'baseColour']]
output_df.to_csv(CATALOG_CSV, index=False)
logger.info(f"Fashion catalog with embeddings saved as {CATALOG_CSV}")
logger.info(f"Catalog contains {len(output_df)} items")