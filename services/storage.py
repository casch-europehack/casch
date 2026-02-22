import json
import os
from typing import Dict, Any

DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)

def save_to_db(file_hash: str, data: Dict[str, Any]):
    with open(os.path.join(DB_DIR, f"{file_hash}.json"), "w") as f:
        json.dump(data, f)

def load_from_db(file_hash: str) -> Dict[str, Any]:
    path = os.path.join(DB_DIR, f"{file_hash}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
