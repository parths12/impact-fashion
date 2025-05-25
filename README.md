# Shopsmarter - AI-Powered Fashion Recommender

Shopsmarter is an intelligent fashion recommendation system that uses AI to provide personalized clothing suggestions based on user preferences, uploaded images, and text queries.

## Features

- **Visual Search**: Upload images to find similar fashion items
- **Text Search**: Search for items using natural language queries
- **Smart Filtering**: Filter by gender and other attributes
- **Shopping Cart**: Add items to cart and checkout
- **AI Assistant**: Interactive chat interface for help and recommendations (Note: Currently provides mock responses without an OpenAI API key)
- **Accessibility**: Built with accessibility in mind

## Dataset

Due to size constraints, the large fashion image dataset and generated catalog files are not included in this repository. To run the application, you will need to:

1.  Download the dataset from [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) on Kaggle.
2.  Extract the downloaded archive.
3.  Inside the extracted files, you should find a `styles.csv` file and an `images` directory containing the image files.
4.  Create a `dataset` directory at the root of this project.
5.  Move the `styles.csv` file and the `images` directory into the newly created `dataset` directory.
6.  Ensure your project structure looks like this:
    ```
    your-project-root/
    ├── app.py
    ├── ... (other project files)
    ├── dataset/
    │   ├── images/  # Contains image files (.jpg)
    │   └── styles.csv # Contains image metadata
    └── ... (other project directories/files)
    ```
7.  Generate the catalog with image embeddings (this will create `catalog.csv`):
    ```bash
    python generate_fashion_catalog.py
    ```

## Technical Stack

- **Backend**: Python/Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **AI/ML**: CLIP for image embeddings, FAISS for similarity search
- **Payment**: Stripe integration (using test keys)
- **Accessibility**: Basic accessibility considerations and potential for further audits (e.g., using browser developer tools, axe-core)

## Setup Instructions

1.  Create a **new** repository on GitHub named `shopsmarter`.
2.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/shopsmarter.git
    cd shopsmarter
    ```
3.  Copy the code from your local project directory into the new `shopsmarter` directory.
4.  Set up the dataset following the instructions in the [Dataset](#dataset) section above.

5.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

6.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

7.  Run the application:
    ```bash
    python app.py
    ```

The application will be available at `http://localhost:5000`

## API Endpoints

-   `POST /recommend/`: Get product recommendations based on uploaded image or text query.
-   `POST /upload_image/`: Upload a new image and save its details to `styles.csv` for potential catalog regeneration.
-   `POST /add_to_cart/`: Add a product to the shopping cart.
-   `GET /cart`: View the shopping cart.
-   `POST /remove_from_cart/`: Remove a product from the shopping cart.
-   `POST /create-payment-intent/`: Create a Stripe payment intent for checkout.
-   `GET /payment-success`: Redirect page after successful payment.
-   `POST /chat`: Interact with the AI assistant (currently mock responses).

## Extending to Other Categories

You can adapt this application for other product categories by replacing the fashion dataset with a dataset for your desired category and ensuring the `generate_fashion_catalog.py` script and `filter_products` function (if used) are compatible with your new data structure.
