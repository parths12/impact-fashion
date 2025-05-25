from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from embedding import get_clip_embedding, get_clip_text_embedding
import faiss
import logging
from nlp_processor import NLPProcessor
import stripe
from flask_session import Session
from detection import detect_clothing_type
from transformers import pipeline
import openai
import re
import uuid

app = Flask(__name__)

# Configure session
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configure Stripe
stripe.api_key = 'STRIPE_KEY'  # Replace with your Stripe secret key
STRIPE_PUBLISHABLE_KEY = 'stripe_key'  # Replace with your Stripe publishable key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the catalog
CATALOG_CSV = 'catalog.csv'
DATASET_DIR = 'dataset/images'
STYLES_CSV = 'dataset/styles.csv'  # Path to styles.csv

def load_catalog():
    logger.info("Loading catalog from CSV...")
    df = pd.read_csv(CATALOG_CSV)
    logger.info(f"Loaded catalog with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample of first few rows:")
    logger.info(df.head().to_string())
    
    # Convert embeddings from string to numpy array
    logger.info("Converting embeddings to numpy arrays...")
    # Ensure the 'embedding' column exists and handle potential NaN values
    if 'embedding' not in df.columns or df['embedding'].isnull().all():
        logger.warning("No embeddings found in catalog.csv.")
        embeddings = np.array([])
    else:
        embeddings = np.array([np.fromstring(emb, sep=',') for emb in df['embedding'].dropna()])

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return df, embeddings

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logger.warning("No embeddings to create FAISS index.")
        return None
    # Normalize embeddings
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    return index

def normalize_term(term):
    """Normalize a search term for better matching."""
    term = term.lower().strip()
    # Add more normalization rules as needed
    return term

def normalize_gender(g):
    """Normalize gender values for consistent filtering."""
    g = str(g).lower().strip()
    if g in ['men', 'male', 'm']:
        return 'men'
    elif g in ['women', 'female', 'f']:
        return 'women'
    elif g in ['unisex', 'u']:
        return 'unisex'
    return g

def filter_products(df, filter_type, text_query=''):
    logger.info(f"Starting filter_products with filter_type={filter_type}, text_query={text_query}")

    filtered_df = df.copy()

    # Normalize gender column once
    filtered_df['gender_normalized'] = filtered_df['gender'].apply(normalize_gender)
    logger.info(f"After gender normalization, unique genders: {filtered_df['gender_normalized'].unique()}")

    # Apply gender/type filter
    if filter_type != 'All':
        if filter_type in ['Men', 'Women']:
            # Ensure 'gender' column exists and handle potential NaN values before applying filter
            if 'gender' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df['gender'].apply(normalize_gender) == filter_type.lower()]
            else:
                 logger.warning("'gender' column not found for filtering.")

        # Remove filtering logic for Casual and Formal
        # elif filter_type in ['Casual', 'Formal']:
        #     # Ensure 'usage' column exists and handle potential NaN values before applying filter
        #     if 'usage' in filtered_df.columns:
        #         filtered_df = filtered_df[filtered_df['usage'] == filter_type]
        #     else:
        #         logger.warning("'usage' column not found for filtering.")

    logger.info(f"After filter type '{filter_type}', shape: {filtered_df.shape}")

    # Apply text query filter with AND logic
    if text_query:
        query_terms = [normalize_term(t) for t in text_query.split()]
        logger.info(f"Normalized query terms: {query_terms}")

        # Ensure necessary columns exist before applying text query filter
        name_exists = 'name' in filtered_df.columns
        type_exists = 'articleType' in filtered_df.columns
        color_exists = 'baseColour' in filtered_df.columns

        if not (name_exists or type_exists or color_exists):
            logger.warning("Necessary columns ('name', 'articleType', 'baseColour') not found for text query filtering.")
            # If no searchable columns exist, return empty dataframe for this filter
            return filtered_df[0:0] # return empty dataframe with same columns

        df_lower_name = filtered_df['name'].astype(str).str.lower() if name_exists else None
        df_lower_type = filtered_df['articleType'].astype(str).str.lower() if type_exists else None
        df_lower_color = filtered_df['baseColour'].astype(str).str.lower() if color_exists else None

        # Start with all True mask
        mask = pd.Series(True, index=filtered_df.index)

        # Apply each term with OR logic
        for term in query_terms:
            term_mask_parts = []
            if name_exists: term_mask_parts.append(df_lower_name.str.contains(term, na=False))
            if type_exists: term_mask_parts.append(df_lower_type.str.contains(term, na=False))
            if color_exists: term_mask_parts.append(df_lower_color.str.contains(term, na=False))

            if term_mask_parts:
                term_mask = term_mask_parts[0]
                for part in term_mask_parts[1:]:
                    term_mask = term_mask | part
                mask = mask & term_mask
                logger.info(f"After applying term '{term}', matches: {term_mask.sum()}")
            else:
                # If no searchable columns, no matches for this term
                mask = mask & False

        filtered_df = filtered_df[mask]
        logger.info(f"After text query filtering, shape: {filtered_df.shape}")

        if len(filtered_df) > 0:
            logger.info("Sample of filtered items:")
            for _, item in filtered_df.head().iterrows():
                logger.info(f"Name: {item.get('name', 'Unknown Product')}, Type: {item.get('articleType', 'unknown')}, Color: {item.get('baseColour', 'unknown')}")

    return filtered_df

# Load catalog and create FAISS index
logger.info("Loading catalog and creating FAISS index...")
df, embeddings = load_catalog()
index = create_faiss_index(embeddings)
nlp_processor = NLPProcessor()
logger.info("Catalog, index, and NLP processor loaded successfully!")

# Initialize cart in session
def init_cart():
    if 'cart' not in session:
        session['cart'] = []

# Configure OpenAI
client = openai.OpenAI(api_key= 'ENTER_OPENAI_API_KEY')

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
        text_query = request.form.get('text_query', '')

        if 'image' in request.files:
            # Handle image upload for recommendation search
            image_file = request.files['image']
            if image_file.filename:
                try:
                    # Process the uploaded image to get embedding
                    img = Image.open(image_file.stream)
                    img = img.convert('RGB')
                    query_embedding = get_clip_embedding(img)

                    # Ensure index is loaded
                    if index is None:
                         logger.error("FAISS index not loaded.")
                         return jsonify({'error': 'Recommendation system not fully initialized.'}), 500

                    # Perform FAISS search using the image embedding
                    # Search the entire catalog (df and index) based on the image embedding
                    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k=5) # Get top 5 similar items
                    
                    # Get the recommended items from the dataframe
                    recommended_items = df.iloc[I[0]]

                    for i, item in recommended_items.iterrows():
                         # Construct image path
                         image_filename = os.path.basename(item["image_path"])
                         image_path = f"/static/images/{image_filename}"
                         logger.info(f"Recommended image path for {item.get('name', 'Unknown Product')}: {image_path}")

                         recommendations.append({
                             'id': str(item['id']),
                             'name': str(item.get('name', 'Unknown Product')),
                             'price': float(item.get('price', 999)), # Use get with default for flexibility
                             'image_path': image_path,
                             'similarity': float(D[0][list(recommended_items.index).index(i)]), # Get similarity score
                             'detected_type': item.get('articleType', 'unknown') # Use get with default
                         })

                except Exception as e:
                    logger.error(f"Error processing image for recommendation: {e}")
                    # Add logging to inspect the problematic item if possible
                    if 'item' in locals():
                         logger.error(f"Problematic item keys: {item.keys()}")
                    return jsonify({'error': str(e)}), 500

        elif text_query:
            # Handle text query and filtering
            filtered_df = filter_products(df, current_filter, text_query)

            if len(filtered_df) == 0:
                return jsonify({'recommendations': []})

            # Get the first 5 items from the filtered results (or fewer if less than 5)
            for _, item in filtered_df.head(5).iterrows():
                # Construct image path
                image_filename = os.path.basename(item["image_path"])
                image_path = f"/static/images/{image_filename}"
                logger.info(f"Image path for {item.get('name', 'Unknown Product')}: {image_path}")

                recommendations.append({
                    'id': str(item['id']),
                    'name': str(item.get('name', 'Unknown Product')),
                    'price': float(item.get('price', 999)), # Use get with default
                    'image_path': image_path,
                    'similarity': 1.0,  # Assume perfect match for filtered text search results
                    'detected_type': item.get('articleType', 'unknown') # Use get with default
                })

        # Note: If neither image nor text query is provided, the initial check returns a 400.
        # If an image is provided, it does image search. If text is provided, it does text search.
        # If both were provided, this current logic would prioritize image search.

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded image
        filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
        image_path = os.path.join(DATASET_DIR, filename)
        image_file.save(image_path)
        logger.info(f"Saved uploaded image to {image_path}")

        # Get image details from form data
        brand_name = request.form.get('brand_name', 'Custom Upload')
        article_type = request.form.get('type', 'unknown')
        color = request.form.get('color', 'unknown')
        gender = request.form.get('gender', 'unisex')
        usage = request.form.get('usage', 'Casual')

        # Read existing styles.csv
        styles_df = pd.read_csv(STYLES_CSV)

        # Create a new row for the uploaded image
        new_row = {
            'id': str(uuid.uuid4()), # Generate a unique ID
            'productDisplayName': brand_name,
            'gender': gender,
            'articleType': article_type,
            'baseColour': color,
            'usage': usage,
            'image_path': filename, # Save just the filename relative to DATASET_DIR
            'embedding': None # Embedding will be added by generate_fashion_catalog.py
        }

        # Append the new row to the DataFrame
        styles_df = pd.concat([styles_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated DataFrame back to styles.csv
        styles_df.to_csv(STYLES_CSV, index=False)

        logger.info(f"Added new entry to {STYLES_CSV} for image {filename}")

        # Note: Catalog regeneration and server restart are needed to include this
        # new item in the recommendations.

        return jsonify({'message': 'Image uploaded and details saved. To include this item in recommendations, please regenerate the catalog (python generate_fashion_catalog.py) and restart the server (python app.py).', 'image_filename': filename}), 200

    except Exception as e:
        logger.error(f"Error uploading image or saving details: {e}")
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

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    try:
        product_id = request.json.get('product_id')
        logger.info(f"Received request to add product to cart. Product ID: {product_id}")

        if not product_id:
            logger.error("No product ID provided in add to cart request.")
            return jsonify({'error': 'No product ID provided'}), 400

        # Convert product_id to string for robust comparison
        product_id_str = str(product_id)

        # Get product details from catalog
        # Ensure 'id' column is treated as string for robust comparison
        product_df = df[df['id'].astype(str) == product_id_str]
        logger.info(f"Found {len(product_df)} matching products in catalog for add to cart.")

        if product_df.empty:
            logger.error(f"Product not found in catalog for add to cart. ID: {product_id}")
            return jsonify({'error': 'Product not found in catalog'}), 404

        product = product_df.iloc[0]
        # Use .get() with a default value for accessing product details to avoid KeyError
        logger.info(f"Product details for add to cart: ID={product.get('id', 'N/A')}, Name={product.get('name', 'Unknown Product')}, Price={product.get('price', 'N/A')}")

        # Initialize cart if not exists
        init_cart()

        # Add product to cart
        cart_item = {
            'id': str(product.get('id', '')),
            'name': str(product.get('name', 'Unknown Product')),
            'price': float(product.get('price', 0.0)),
            'image_path': url_for('static', filename=f'images/{os.path.basename(product.get('image_path', ''))}'),
            'quantity': 1
        }

        # Check if product already in cart
        for item in session['cart']:
            if str(item['id']) == product_id_str:
                item['quantity'] += 1
                session.modified = True
                logger.info(f"Updated quantity for product {product_id} in cart.")
                return jsonify({'message': 'Product quantity updated in cart', 'cart': session['cart']})

        session['cart'].append(cart_item)
        session.modified = True
        logger.info(f"Added new product {product_id} to cart.")

        return jsonify({'message': 'Product added to cart', 'cart': session['cart']})

    except Exception as e:
        logger.error(f"Error adding to cart: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cart')
def view_cart():
    init_cart()
    cart = session['cart']
    total = sum(item['price'] * item['quantity'] for item in cart)
    return render_template('cart.html', 
                         cart=cart,
                         total=total,
                         stripe_public_key=STRIPE_PUBLISHABLE_KEY)

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    try:
        product_id = request.json.get('product_id')
        if not product_id:
            return jsonify({'error': 'No product ID provided'}), 400
        
        init_cart()
        
        # Remove product from cart
        session['cart'] = [item for item in session['cart'] if item['id'] != product_id]
        session.modified = True
        
        return jsonify({'message': 'Product removed from cart', 'cart': session['cart']})
    
    except Exception as e:
        logger.error(f"Error removing from cart: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create-payment-intent', methods=['POST'])
def create_payment_intent():
    try:
        init_cart()
        
        # Calculate total amount
        total_amount = sum(item['price'] * item['quantity'] for item in session['cart'])
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=int(total_amount * 100),  # Convert to cents
            currency='usd'
        )
        
        return jsonify({
            'clientSecret': intent.client_secret
        })
    
    except Exception as e:
        logger.error(f"Error creating payment intent: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/payment-success')
def payment_success():
    # Clear cart after successful payment
    session['cart'] = []
    session.modified = True
    return render_template('payment_success.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': "Please enter a message."})

    logger.info(f"Received chat message (using customized mock): {user_message}")

    # --- Start Customized Mock Assistant Logic ---
    user_message_lower = user_message.lower()
    reply = "Thank you for your message. This is a mock response. The AI assistant is not active without a valid API key."

    if any(word in user_message_lower for word in ["hello", "hi", "hey"]):
        reply = "Hello there! How can I help you today?"
    elif any(word in user_message_lower for word in ["product", "item", "recommend"]):
        reply = "You can find products by using the search bar above or by uploading an image for visual recommendations."
    elif any(word in user_message_lower for word in ["cart", "bag", "checkout"]):
        reply = "You can view your cart by clicking the shopping cart icon in the top right corner."
    elif any(word in user_message_lower for word in ["help", "support", "assist"]):
        reply = "I am a mock AI assistant for this fashion recommender demo. You can try searching for products or uploading an image."
    elif "mock" in user_message_lower:
         reply = "Yes, I am currently providing mock responses as the real AI assistant is not active."

    # --- End Customized Mock Assistant Logic ---

    logger.info(f"Sending customized mock chat response: {reply}")

    # --- Original OpenAI API Logic (Commented Out) ---
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful AI assistant for a fashion e-commerce site. You can answer questions, help with product search, and assist with cart and checkout."},
    #             {"role": "user", "content": user_message}
    #         ]
    #     )
    #     reply = response.choices[0].message.content
    # except Exception as e:
    #     reply = f"Sorry, there was an error contacting the AI assistant: {e}"
    # --- End Original OpenAI API Logic ---

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
