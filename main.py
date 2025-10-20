import os
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
import torch
from gliner import GLiNER

from dotenv import load_dotenv
load_dotenv()  # reads .env file

import os
token = os.getenv("HUGGINGFACE_HUB_TOKEN")


model_name = "gamatoad/inflation-narrative"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


model_gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
model_edges = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
app = FastAPI()


origins = [
    "http://localhost:5173",
    "https://colinfrugte.github.io/infl_narrative/"  
]

# 👇 Middleware aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # oder ["*"] für alle
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],  
    allow_headers=["*"],
)

class NodeScore(BaseModel):
    id: str
    score: float

class EdgeScore(BaseModel):
    source: str
    target: str
    score: float

class FullScoreResponse(BaseModel):
    nodes: List[NodeScore]
    edges: List[EdgeScore]

class Message(BaseModel):
    text: str

@app.post("/scores", response_model=FullScoreResponse)
def get_scores(message: Message):
    node_scores_dict = predict_nodes(message.text)

    node_entries = [
        NodeScore(id=label, score=score)
        for label, score in node_scores_dict.items()
    ]
    edge_entries = predict_edges(node_scores_dict, message.text)

    return {
        "nodes": node_entries,
        "edges": edge_entries
    }


def predict_nodes(text, threshold=0.2):
    # Modell in den Evaluationsmodus versetzen
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Logits extrahieren und Sigmoid anwenden
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    scores = probs.squeeze().tolist()

        # Label-Namen laden
    try:
        label_names = model.config.id2label
    except AttributeError:
        label_names = {i: f"Label_{i}" for i in range(len(scores))}

    # Dictionary mit Labeln & Scores
    label_scores = {label_names[i]: float(score) for i, score in enumerate(scores)}
    
    #############
    #############
    #############
    #### this is only for testing the gliner model. 
    #############
    # just for testing what the gliner does.
    label_scores = {label: 0.0 for label in label_names.values()}
    # gliner model and scores appending
    entities = model_gliner.predict_entities(text, label_names.values())
    # updaten der label scores mit denen vom gliner
    for entity in entities:
        label_scores[entity["label"]] = entity["score"]
    ##########
    ##########
    ##########
    ##########

    # appending inflation as final label
    label_scores["inflation"] = 1.0  # falls gewünscht

    return label_scores

def predict_edges(node_score_dict: dict, text: str) -> List[EdgeScore]:
    labels = list(node_score_dict.keys())

    node_combinations = []
    for i, source in enumerate(labels):
        if source == "inflation":
            continue
        for j, target in enumerate(labels):
            if i != j:
                node_combinations.append({
                    "source": source,
                    "target": target
                })

    combinations_as_text = [
        (text, f'{combo["source"]} is leading to {combo["target"]}')
        for combo in node_combinations
    ]

    scores = model_edges.predict(combinations_as_text)

    return [
        EdgeScore(source=combo["source"], target=combo["target"], score=float(score))
        for combo, score in zip(node_combinations, scores)
    ]