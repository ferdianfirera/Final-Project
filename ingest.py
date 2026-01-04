# ingest.py
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Load env vars
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "olist_docs")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CSV_FILE = os.getenv("CSV_FILE", "data_newmerge.csv")

# Validate key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Qdrant Client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Vector size for embedding 3-small
VECTOR_SIZE = 1536

# Create collection if not exist
collections = qdrant.get_collections().collections
existing = [c.name for c in collections]

if QDRANT_COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE, 
            distance=Distance.COSINE,
            on_disk=True  # <--- penting untuk dataset besar 100k+
        ),
    )
    print(f"Created collection: {QDRANT_COLLECTION}")
else:
    print(f"Collection '{QDRANT_COLLECTION}' already exists.")

# Check CSV
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

# Load
df = pd.read_csv(CSV_FILE)

# Convert row -> text
def row_to_text(row) -> str:
    text = (
        f"Order {row.get('order_id')} by customer {row.get('customer_id')} located in "
        f"{row.get('customer_city')}, {row.get('customer_state')}.\n"
        f"Product: {row.get('product_id')}, Category: {row.get('product_category_name_english')}.\n"
        f"Price: {row.get('price')}, Freight: {row.get('freight_value')}.\n"
        f"Review score: {row.get('review_score')}. Comment: {row.get('review_comment_message')}.\n"
        f"Seller: {row.get('seller_id')} ({row.get('seller_city')}, {row.get('seller_state')}).\n"
        f"Payment: {row.get('payment_type')}.\n"
        f"Product details: {row.get('product_weight_g')}g, "
        f"{row.get('product_length_cm')}x{row.get('product_height_cm')}x{row.get('product_width_cm')} cm."
    )
    return text[:2000]  # truncate → hemat token

# Batch sizes
BATCH_EMBED = 200       # embedding 200 per call
BATCH_UPSERT = 200      # upsert 200 vectors per call

texts = []
payloads = []
ids = []

print("\nStarting ingestion...\n")

for i, row in tqdm(df.iterrows(), total=len(df)):

    txt = row_to_text(row)
    texts.append(txt)
    payloads.append(row.to_dict() | {"text": txt})
    ids.append(str(uuid.uuid4()))

    # If batch ready → embed + upload
    if len(texts) >= BATCH_EMBED:
        # ---- EMBEDDING ----
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        vectors = [d.embedding for d in response.data]

        # Build points
        points = [
            {
                "id": ids[j],
                "vector": vectors[j],
                "payload": payloads[j],
            }
            for j in range(len(vectors))
        ]

        # ---- UPSERT TO QDRANT ----
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

        texts, payloads, ids = [], [], []

# Flush sisa batch
if texts:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in response.data]

    points = [
        {"id": ids[j], "vector": vectors[j], "payload": payloads[j]}
        for j in range(len(vectors))
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

print(f"\nIngestion COMPLETE — total rows: {len(df)}")
