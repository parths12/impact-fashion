import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract key features from text."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract key terms
        tokens = word_tokenize(processed_text)
        term_freq = Counter(tokens)
        
        # Extract common fashion-related terms
        fashion_terms = {
            'style': ['casual', 'formal', 'sporty', 'elegant', 'vintage'],
            'color': ['red', 'blue', 'black', 'white', 'green', 'yellow'],
            'material': ['cotton', 'denim', 'leather', 'silk', 'wool'],
            'pattern': ['striped', 'floral', 'plaid', 'solid', 'print'],
            'type': ['shirt', 'dress', 'pants', 'jacket', 'skirt']
        }
        
        features = {}
        for category, terms in fashion_terms.items():
            features[category] = [term for term in terms if term in processed_text]
        
        return features
    
    def compute_text_similarity(self, text1, text2):
        """Compute similarity between two text descriptions."""
        # Preprocess texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def analyze_product_description(self, description):
        """Analyze product description and extract key information."""
        # Preprocess text
        processed_text = self.preprocess_text(description)
        
        # Extract features
        features = self.extract_features(processed_text)
        
        # Extract price if present
        price_pattern = r'\$\d+(\.\d{2})?'
        price_match = re.search(price_pattern, description)
        price = float(price_match.group(0).replace('$', '')) if price_match else None
        
        # Extract size if present
        size_pattern = r'\b(XS|S|M|L|XL|XXL)\b'
        size_match = re.search(size_pattern, description)
        size = size_match.group(0) if size_match else None
        
        return {
            'features': features,
            'price': price,
            'size': size,
            'processed_text': processed_text
        }
    
    def get_recommendations(self, query_text, product_descriptions, top_k=5):
        """Get text-based recommendations based on similarity."""
        # Preprocess query
        processed_query = self.preprocess_text(query_text)
        
        # Create TF-IDF vectors for all descriptions
        all_texts = [processed_query] + [self.preprocess_text(desc) for desc in product_descriptions]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Compute similarities
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        # Get top-k recommendations
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices] 