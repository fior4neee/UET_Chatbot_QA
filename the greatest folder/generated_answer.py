from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

es = Elasticsearch(['https://localhost:9200'], http_auth=('elastic', '*VeQpz+N*KTvpBHhIGE-'), verify_certs=False)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

query = "Phương thức xét tuyển ưu tiên ở trường UET là gì?"
query_embedding = get_embedding(query)

response = es.search(index="pdf_embeddings", body={
    "query": {
        "match_all": {}
    },
    "_source": ["text", "text_embeddings"]  
})

document_embeddings = [doc['_source']['text_embeddings'] for doc in response['hits']['hits']]

query_embedding = np.array(query_embedding).reshape(1, -1)  # Ensure query embedding is <=2
document_embeddings = np.array(document_embeddings)

# Check if document_embeddings is 3D and reshape if necessary
if document_embeddings.ndim == 3:
    document_embeddings = document_embeddings.reshape(document_embeddings.shape[0], -1)
similarities = cosine_similarity(query_embedding, document_embeddings)

most_similar_document = response['hits']['hits'][np.argmax(similarities)]

print("Most similar documents:", most_similar_document['_source']['text'])

# Combine the context (text from most similar documents) and the query to feed to the language model
context = most_similar_document['_source']['text']

combined_input = query + " " + context

# I HAVENT CHUNKED YET

# Combined input length is within the model's max token limit
inputs = tokenizer(combined_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
print(f"Inputs: {inputs}")  # Debugging the tokenized input
# print(f"Tokenized input length: {len(inputs['input_ids'][0])}") # 512 even though mine can be 768, i truncated it for phobert
# Get the answer from the model
with torch.no_grad():
    outputs = model(**inputs)

print("Answer:", outputs)
