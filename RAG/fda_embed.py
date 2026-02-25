import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json
import shutil
from tqdm import tqdm
import chromadb
import hashlib
from huggingface_hub import login
from chromadb.utils import embedding_functions

import wandb

wandb.init(mode="disabled")
os.environ["WANDB_DISABLED"] = "true"

hf_token = os.getenv("hf_token")
login(hf_token)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="google/embeddinggemma-300m", device="cuda")

client = chromadb.PersistentClient(path="./fda_db")
collection = client.get_or_create_collection(name="text_chunks", embedding_function=sentence_transformer_ef)

BATCH_SIZE = 500

for n in range(1, 14):
    with open("drug-label-00{str(n).zfill(2)}-of-0013.json") as f:
        fda_data = json.load(f)["results"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", "}", "]", "}", ",", ". ", " ", ""],
            length_function=len,
        )

        all_ids = []
        all_docs = []
        all_metas = []

        for prod_idx, product_json in enumerate(tqdm(fda_data[:], desc="Processing products")):
            try:
                product_id = product_json["openfda"]["brand_name"][0]
            except:
                try:
                    product_id = product_json["active_ingredient"][0]
                except:
                    continue

            for field_name, value in product_json.items():
                if not isinstance(value, list):
                    value = [str(value)]

                for item_idx, text in enumerate(value):
                    if not text or len(text.strip()) < 20:
                        continue

                    chunks = splitter.split_text(text)

                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = f"{product_id}_{field_name}_{item_idx}_{chunk_idx}"

                        if chunk_id in all_ids:
                            continue

                        all_ids.append(chunk_id)
                        all_docs.append(chunk)
                        all_metas.append(
                            {
                                "product_id": product_id,
                                "field": field_name,
                                "item_idx": item_idx,
                                "chunk_idx": chunk_idx,
                                "original_prod_idx": prod_idx,
                            }
                        )

        for i in tqdm(range(0, len(all_ids), BATCH_SIZE)):
            batch_ids = all_ids[i : i + BATCH_SIZE]
            batch_docs = all_docs[i : i + BATCH_SIZE]
            batch_metas = all_metas[i : i + BATCH_SIZE]

            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
