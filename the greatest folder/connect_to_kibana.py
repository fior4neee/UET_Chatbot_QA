from elasticsearch import Elasticsearch
import pandas as pd
import ast
import numpy as np



# Connect to Elasticsearch
es = Elasticsearch(
    ['https://localhost:9200'],  # Full URL with the scheme
    http_auth=('elastic', '*VeQpz+N*KTvpBHhIGE-'),
    verify_certs=False  # Set to True if SSL certificates are properly configured
)

# Read the data from CSV
all_pages_and_texts_df = pd.read_csv("D:/VSCC/Python K7/all_pages_and_texts.csv")
# print(all_pages_and_texts_df.head())
all_pages_and_texts_df.fillna('', inplace=True)  # Replace NaNs with empty strings

all_pages_and_texts_df['page_number'] = pd.to_numeric(all_pages_and_texts_df['page_number'], errors='coerce').fillna(0).astype(int)
all_pages_and_texts_df['page_char_count'] = pd.to_numeric(all_pages_and_texts_df['page_char_count'], errors='coerce').fillna(0).astype(int)
all_pages_and_texts_df['page_word_count'] = pd.to_numeric(all_pages_and_texts_df['page_word_count'], errors='coerce').fillna(0).astype(int)

all_pages_and_texts_df['text_embeddings'] = all_pages_and_texts_df['text_embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


for index, row in all_pages_and_texts_df.iterrows():
    document = {
        "page_number": row["page_number"],
        "page_from": row["page_from"],
        "page_char_count": row["page_char_count"],
        "page_word_count": row["page_word_count"],
        "page_sentence_count": row["page_sentence_count"],
        "page_token_count": row["page_token_count"],
        "text": row["text"],
        "text_embeddings": row["text_embeddings"]
    }
    
    try:
        response = es.index(index="pdf_embeddings", id=index, document=document)
        print(f"Indexed document {index}: {response['result']}")
    except Exception as e:
        print(f"Failed to index document {index}: {str(e)}")
