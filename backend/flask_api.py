import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, pipeline
from huggingface_hub import login
from utils import *
import os

os.environ["WANDB_DISABLED"] = "true"

TEST = True

hf_token = os.getenv("hf_token")
login(hf_token)

model_id = "google/medgemma-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

role_instruction = "You are an expert doctor."

system_instruction = f"""SYSTEM INSTRUCTION: {role_instruction}. Answer with reasoning."""

sysmsg = {"role": "system", "content": [{"type": "text", "text": system_instruction}]}

app = Flask(__name__)
CORS(app)

cache = {}


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers["X-Content-Type-Options"] = "*"
        return res


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/patientdetails", methods=["GET"])
def get_patient_details():
    if TEST:
        with open("medication.json") as f:
            details = json.load(f)
        cache["patientdetails"] = details
    else:
        pid = request.args.get("pid", PATIENT_ID)
        token = get_access_token()
        details = get_patient_medications(token, pid)
    return _corsify_actual_response(jsonify(details))


@app.route("/run-ner", methods=["GET"])
def run_ner():
    pid = request.args.get("pid")
    print(pid)
    with open("medication.json") as f:
        js = json.load(f)
    return _corsify_actual_response(jsonify(js))


@app.route("/processpatient", methods=["POST"])
def processpatient():

    response = "The patient is currently on YAZ (drospirenone-ethinyl estradiol). Metformin is generally safe, but there is a potential for increased risk of lactic acidosis, especially in patients with renal impairment, liver disease, or those who are dehydrated. While not a direct interaction, the combination of YAZ and metformin could potentially increase the risk of these adverse effects. The patient's FBG is elevated, which could be a sign of underlying metabolic issues that could also increase the risk of lactic acidosis. Therefore, careful monitoring is warranted."
    return _corsify_actual_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
