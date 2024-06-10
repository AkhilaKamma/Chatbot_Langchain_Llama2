from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
import uuid, time
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')



extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY  
)

spec = ServerlessSpec(cloud = "aws", region=PINECONE_API_ENV)

index_name="medical-bot"

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of embed 3
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
# time.sleep(1)
# # view index stats
# index.describe_index_stats()

def store_embeddings_in_pinecone(text_chunks, embeddings):
    for chunk in text_chunks:
        # Generate embeddings for each chunk
        chunk_text = chunk.page_content
        embedding = embeddings.embed_documents([chunk_text])[0]
        
        # Generate a unique ID for the chunk
        chunk_id = str(uuid.uuid4())
        
        # Upsert the embedding and chunk into Pinecone
        index.upsert([(chunk_id, embedding, {"text": chunk_text})])
store_embeddings_in_pinecone(text_chunks, embeddings)