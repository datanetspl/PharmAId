import os
import uuid
import time
import jwt
import requests

CLIENT_ID = os.getenv("CLIENT_ID")
PRIVATE_KEY_PATH = "private_key.pem"
kid = os.getenv("kid")

TOKEN_URL = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
BASE_URL = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"

# Specific test patient ID (Camila Lopez)
PATIENT_ID = "erXuFYUfucBZaryVksYEcMg3"


def get_access_token():
    """Generates a signed JWT and exchanges it for an Epic Access Token."""
    with open(PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    # Epic requires 'exp' to be no more than 5 minutes in the future
    payload = {"iss": CLIENT_ID, "sub": CLIENT_ID, "aud": TOKEN_URL, "jti": str(uuid.uuid4()), "iat": int(time.time()), "exp": int(time.time()) + 280}

    headers = {"alg": "RS384", "typ": "JWT", "kid": kid}

    # Encode the JWT using RS384 (Epic standard)
    signed_jwt = jwt.encode(payload, private_key, algorithm="RS384", headers=headers)

    # Exchange JWT for Access Token
    payload = {"grant_type": "client_credentials", "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer", "client_assertion": signed_jwt, "scope": "system/Patient.read system/MedicationRequest.read"}

    response = requests.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")


def get_patient_medications(token, patient_id):
    """Fetches all MedicationRequest resources for a specific patient."""
    # Epic requires the 'patient' or 'subject' parameter for searches
    url = f"{BASE_URL}/MedicationRequest"
    params = {"patient": patient_id}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()
