from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize Elasticsearch connection
es = Elasticsearch(['https://localhost:9200'], http_auth=('elastic', '*VeQpz+N*KTvpBHhIGE-'), verify_certs=False)

# Load the Vietnamese model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Function to get the embeddings of the question
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Query to search Elasticsearch for similar documents
query = "Phương thức xét tuyển ưu tiên ở trường UET là gì?"
query_embedding = get_embedding(query)

# Elasticsearch query to fetch document embeddings (replace `pdf_embeddings` with your index name)
response = es.search(index="pdf_embeddings", body={
    "query": {
        "match_all": {}
    },
    "_source": ["text", "text_embeddings"]  # Make sure to include the text and embeddings
})

# Extract document embeddings from Elasticsearch response
document_embeddings = [doc['_source']['text_embeddings'] for doc in response['hits']['hits']]

# Reshape query and document embeddings to 2D arrays for cosine similarity
query_embedding = np.array(query_embedding).reshape(1, -1)  # Ensure query embedding is 2D
document_embeddings = np.array(document_embeddings)

# Check if document_embeddings is 3D and reshape if necessary
if document_embeddings.ndim == 3:
    document_embeddings = document_embeddings.reshape(document_embeddings.shape[0], -1)

# Calculate the cosine similarity between the query and each document's embedding
similarities = cosine_similarity(query_embedding, document_embeddings)

# Get the most similar document
most_similar_document = response['hits']['hits'][np.argmax(similarities)]

# Print the most similar document
# print("Most similar documents:", most_similar_document['_source']['text'])

# Combine the context (text from most similar documents) and the query to feed to the language model
context = most_similar_document['_source']['text']

combined_input = query + " " + context

# Ensure the combined input length is within the model's max token limit
inputs = tokenizer(combined_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
# print(f"Inputs: {inputs}")  # Debugging the tokenized input
print(f"Tokenized input length: {len(inputs['input_ids'][0])}") # 512 even though mine can be 768, i truncated it for phobert
# Get the answer from the model
with torch.no_grad():
    outputs = model(**inputs)

# Generate the answer (e.g., using a generation model or a classifier, depending on your setup)
print("Answer:", outputs)
