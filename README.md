# AI Visual Product Recommender Backend

## Category: Electronics (Demo)

This backend demonstrates a visual recommendation system for electronics products using object detection (YOLOv8), image embeddings (CLIP), and fast similarity search (FAISS).

- **Sample Data:** The product catalog is a small, curated set of electronics items with demo images and randomly generated embeddings for hackathon/demo purposes.
- **No real user/product data is used.**

## How to Run

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Generate the demo catalog:
   ```
   python generate_catalog.py
   ```
3. Start the backend server:
   ```
   uvicorn app:app --reload
   ```

## API Endpoints
- `POST /analyze/` — Upload an image, get number of detected objects.
- `POST /recommend/` — Upload the same image and specify `object_index` (0-based) to get top-5 similar electronics products.

---
**You can extend this backend to other categories by updating the catalog and embeddings.** 