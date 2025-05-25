import pandas as pd
import numpy as np

# Example electronics products
products = [
    {"id": "phone_1", "name": "Smartphone X", "price": 699, "image_url": "https://dummyimage.com/200x200/000/fff&text=Phone+X"},
    {"id": "laptop_1", "name": "Laptop Pro", "price": 1299, "image_url": "https://dummyimage.com/200x200/000/fff&text=Laptop+Pro"},
    {"id": "headphones_1", "name": "Wireless Headphones", "price": 199, "image_url": "https://dummyimage.com/200x200/000/fff&text=Headphones"},
    {"id": "tablet_1", "name": "Tablet S", "price": 499, "image_url": "https://dummyimage.com/200x200/000/fff&text=Tablet+S"},
    {"id": "camera_1", "name": "Digital Camera", "price": 599, "image_url": "https://dummyimage.com/200x200/000/fff&text=Camera"},
    {"id": "smartwatch_1", "name": "Smart Watch", "price": 299, "image_url": "https://dummyimage.com/200x200/000/fff&text=Watch"},
    {"id": "speaker_1", "name": "Bluetooth Speaker", "price": 99, "image_url": "https://dummyimage.com/200x200/000/fff&text=Speaker"},
    {"id": "monitor_1", "name": "4K Monitor", "price": 399, "image_url": "https://dummyimage.com/200x200/000/fff&text=Monitor"},
    {"id": "router_1", "name": "WiFi Router", "price": 89, "image_url": "https://dummyimage.com/200x200/000/fff&text=Router"},
    {"id": "console_1", "name": "Game Console", "price": 499, "image_url": "https://dummyimage.com/200x200/000/fff&text=Console"},
]

# Generate random 512D embeddings for demo
for p in products:
    emb = np.random.randn(512)
    emb = emb / np.linalg.norm(emb)
    p["embedding"] = ",".join(map(str, emb))

df = pd.DataFrame(products)
df.to_csv("catalog.csv", index=False)
print("Demo electronics catalog generated as catalog.csv") 