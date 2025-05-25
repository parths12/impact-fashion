from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from embedding import get_clip_embedding
import faiss
import logging
from nlp_processor import NLPProcessor

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the catalog
CATALOG_CSV = 'catalog.csv'
DATASET_DIR = 'dataset/images'

def load_catalog():
    df = pd.read_csv(CATALOG_CSV)
    # Convert embeddings from string to numpy array
    embeddings = np.array([np.fromstring(emb, sep=',') for emb in df['embedding']])
    return df, embeddings

def create_faiss_index(embeddings):
    # Normalize embeddings
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    return index

def filter_products(df, filter_type):
    """Filter products based on category."""
    if filter_type == 'All':
        return df
    
    # Convert product names to lowercase for case-insensitive matching
    df_lower = df.copy()
    df_lower['name_lower'] = df_lower['name'].str.lower()
    
    if filter_type == 'Men':
        return df[df_lower['name_lower'].str.contains('men|male|guy|boy', na=False)]
    elif filter_type == 'Women':
        return df[df_lower['name_lower'].str.contains('women|female|girl|lady', na=False)]
    elif filter_type == 'Casual':
        return df[df_lower['name_lower'].str.contains('casual|everyday|street|comfort', na=False)]
    elif filter_type == 'Formal':
        return df[df_lower['name_lower'].str.contains('formal|business|office|professional', na=False)]
    
    return df

# Load catalog and create FAISS index
logger.info("Loading catalog and creating FAISS index...")
df, embeddings = load_catalog()
index = create_faiss_index(embeddings)
nlp_processor = NLPProcessor()
logger.info("Catalog, index, and NLP processor loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files and not request.form.get('text_query'):
        return jsonify({'error': 'No image or text query provided'}), 400
    
    try:
        recommendations = []
        current_filter = request.form.get('filter', 'All')
        
        # Apply category filter
        filtered_df = filter_products(df, current_filter)
        if filtered_df.empty:
            return jsonify({'recommendations': []})
        
        # Handle image-based search
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                # Read and process the image
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Get embedding for the uploaded image
                query_embedding = get_clip_embedding(img)
                query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                
                # Search for similar items
                k = 5  # Number of recommendations
                distances, indices = index.search(query_embedding, k)
                
                # Get recommendations from filtered dataset
                for idx, distance in zip(indices[0], distances[0]):
                    if idx < len(filtered_df):
                        item = filtered_df.iloc[idx]
                        recommendations.append({
                            'id': str(item['id']),
                            'name': str(item['name']),
                            'price': float(item['price']),
                            'image_path': str(item['image_path']),
                            'similarity': float(distance)
                        })
        
        # Handle text-based search
        text_query = request.form.get('text_query')
        if text_query and text_query.strip():
            # Get text-based recommendations
            product_descriptions = filtered_df['name'].tolist()
            text_recommendations = nlp_processor.get_recommendations(
                text_query, 
                product_descriptions
            )
            
            # Add text-based recommendations
            for idx, similarity in text_recommendations:
                if idx < len(filtered_df):
                    item = filtered_df.iloc[idx]
                    recommendations.append({
                        'id': str(item['id']),
                        'name': str(item['name']),
                        'price': float(item['price']),
                        'image_path': str(item['image_path']),
                        'similarity': float(similarity)
                    })
        # If both image and text query are empty or invalid, return empty recommendations
        if not recommendations:
            return jsonify({'recommendations': []})
        
        # Sort recommendations by similarity and remove duplicates
        recommendations = sorted(
            {r['id']: r for r in recommendations}.values(),
            key=lambda x: x['similarity'],
            reverse=True
        )[:5]
        
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze product description and return features."""
    try:
        description = request.json.get('description', '')
        if not description:
            return jsonify({'error': 'No description provided'}), 400
        
        analysis = nlp_processor.analyze_product_description(description)
        return jsonify(analysis)
    
    except Exception as e:
        logger.error(f"Error analyzing description: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 