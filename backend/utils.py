# Import necessary libraries
import os
import uuid
import time
import jwt
import requests
import sys

# Add the RAG directory to the system path to allow for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG")))
from elasticsearch_indexer import ElasticsearchDrugIndexer
import chromadb

# Get client ID and private key path from environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
PRIVATE_KEY_PATH = "private_key.pem"
kid = os.getenv("kid")

# Define URLs for Epic FHIR API
TOKEN_URL = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
BASE_URL = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"

# Specific test patient ID
PATIENT_ID = "erXuFYUfucBZaryVksYEcMg3"


def get_access_token():
    """Generates a signed JWT and exchanges it for an Epic Access Token."""
    with open(PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    # Create the JWT payload
    payload = {"iss": CLIENT_ID, "sub": CLIENT_ID, "aud": TOKEN_URL, "jti": str(uuid.uuid4()), "iat": int(time.time()), "exp": int(time.time()) + 280}

    headers = {"alg": "RS384", "typ": "JWT", "kid": kid}

    # Encode the JWT
    signed_jwt = jwt.encode(payload, private_key, algorithm="RS384", headers=headers)

    # Exchange JWT for an Access Token
    payload = {"grant_type": "client_credentials", "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer", "client_assertion": signed_jwt, "scope": "system/Patient.read system/MedicationRequest.read"}

    response = requests.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")


def get_patient_medications(token, patient_id):
    """Fetches all MedicationRequest resources for a specific patient from the Epic FHIR API."""
    url = f"{BASE_URL}/MedicationRequest"
    params = {"patient": patient_id}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def basedrugname(drugname):
    """A placeholder function to process a drug name."""
    return drugname


def retrieve_RAG(drugname):
    """Retrieves information about a drug from Elasticsearch and ChromaDB."""
    drugname = basedrugname(drugname)
    
    # Search Elasticsearch for the drug
    es = ElasticsearchDrugIndexer()
    es.connect()
    is_valid, msg, es_count = es.validate_index()
    if not is_valid:
        print("Skipping Elasticsearch phase.")
        es.close()
    docmatches = es.search(drugname)
    
    # Query ChromaDB for the drug
    chroma_client = chromadb.PersistentClient(path="./fda_db")
    collection = chroma_client.get_collection(name="text_chunks")
    all_documents = collection.get(include=["documents", "metadatas", "ids"])
    all_ids = all_documents["ids"]
    filtered_ids = [x for x in all_ids if drugname in x]
    results = collection.query(ids=filtered_ids, n_results=10)
    
    return results