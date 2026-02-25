# Import necessary libraries
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, pipeline, AutoModelForTokenClassification
from huggingface_hub import login
from utils import *
import os

# Suppress wandb warnings
os.environ["WANDB_DISABLED"] = "true"

# Configuration flags
TEST = True
RAG = False

# Hugging Face authentication
hf_token = os.getenv("hf_token")
login(hf_token)

# Load MedGemma model and tokenizer
model_id = "google/medgemma-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load NER model and tokenizer
ner_model_id = "tner/xlm-roberta-base-bc5cdr"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_id)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="average")

# Set device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Define role and system instructions for the model
role_instruction = "You are an expert doctor."
system_instruction = f"""SYSTEM INSTRUCTION: {role_instruction}. Answer with reasoning."""
sysmsg = {"role": "system", "content": [{"type": "text", "text": system_instruction}]}

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# In-memory cache
cache = {}


# Handle preflight requests for CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers["X-Content-Type-Options"] = "*"
        return res


# Add CORS headers to responses
def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response


# Root endpoint
@app.route("/")
def hello():
    return "Hello, World!"


# Endpoint to get patient details
@app.route("/patientdetails", methods=["GET"])
def get_patient_details():
    if TEST:
        # Load patient details from a local JSON file in test mode
        with open("medication.json") as f:
            details = json.load(f)
        cache["patientdetails"] = details
    else:
        # Fetch patient details from Epic API in production mode
        pid = request.args.get("pid", PATIENT_ID)
        token = get_access_token()
        details = get_patient_medications(token, pid)
    return _corsify_actual_response(jsonify(details))


# Endpoint to run NER on patient data
@app.route("/run-ner", methods=["GET"])
def run_ner():
    pid = request.args.get("pid")
    with open("medication.json") as f:
        js = json.load(f)
    return _corsify_actual_response(jsonify(js))


# Endpoint to process patient data and get AI-powered analysis
@app.route("/processpatient", methods=["POST"])
def processpatient():
    data = request.get_json()
    print(data["existing_medication"])
    patientdata = data["existing_medication"]
    newmed = data["new_medication"]
    newsymptoms = data["new_symptoms"]

    # Create a prompt for the model
    prompt = f"""
        The patient is currently on - {patientdata}

        The proposed prescription is {newmed} for {newsymptoms}. Are there any potential  drug-drug interactions, or existing conditions that could cause adverse reactions.
                        
        """
    if RAG:
        # Retrieve additional information from the RAG pipeline if enabled
        addl_prompt = retrieve_RAG(newmed)
        prompt += f"""
            FDA label data - {addl_prompt}
        """

    # Prepare messages for the model
    newmsg = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }
    messages = [sysmsg, newmsg]

    # Process the messages and generate a response
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)

    response = processor.decode(generation, skip_special_tokens=True)

    # Return the model's response
    return _corsify_actual_response(jsonify(response[0][response[0].find("\nmodel") + 7 :]))


# Run the Flask app
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
