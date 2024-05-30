from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("./data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name="medical-bot"

# connect to index
index = pc.Index(index_name)

#Initializing the Pinecone
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
model

#Creating Embeddings for Each of The Text Chunks & storing
vectors = []
for i, embedding in enumerate(text_chunks):    
    vectors.append({
        "id": f"doc_{i}",
        "values": model.encode(embedding.page_content),
        "metadata": {"text": embedding.page_content}
    })

#Creating Embeddings for Each of# Upsert documents into the Pinecone index
for vec in vectors:
    index.upsert([vec])
