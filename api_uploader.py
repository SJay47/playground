# File: api_uploader.py
import requests
import json
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Optional, Dict

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and Pandas data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_config(config_path: Path) -> Dict:
    print(f"Loading API configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"    ✗ ERROR: Could not load or parse config file. {e}"); return None

def get_access_token(config: Dict) -> Optional[str]:
    payload = {"client_id": config["KEYCLOAK_CLIENT_ID"], "grant_type": "password", "username": config["ADMIN_USERNAME"], "password": config["ADMIN_PASSWORD"]}
    try:
        print("🔑 Authenticating with Keycloak...")
        r = requests.post(config["KEYCLOAK_TOKEN_URL"], data=payload, timeout=10)
        r.raise_for_status()
        print("    ✓ Authentication successful.")
        return r.json().get("access_token")
    except requests.RequestException as e:
        print(f"    ✗ Authentication error: {e}"); return None

def create_organization_via_api(config: Dict, token: str, org_name: str) -> Optional[str]:
    url = f"{config['API_BASE_URL']}{config['CREATE_ORGANIZATION_ENDPOINT']}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"data": {"name": org_name, "visibility": "Public", "visibleTo": []}}
    timeout = config.get("API_TIMEOUT", 60) # Use configured timeout
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        org_id = r.json().get("data", {}).get("id")
        if org_id: print(f"    ✓ Organization '{org_name}' created with ID: {org_id}"); return org_id
        else: print(f"    ✗ Org creation for '{org_name}' succeeded but no ID was returned."); return None
    except requests.RequestException as e:
        print(f"    ✗ Failed to create organization '{org_name}': {e}"); return None

def create_dataset_via_api(config: Dict, token: str, org_id: str, dataset_name: str, description: str) -> Optional[str]:
    endpoint = config['CREATE_DATASET_ENDPOINT_TEMPLATE'].format(org_id=org_id)
    url = f"{config['API_BASE_URL']}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"data": {"name": dataset_name, "description": description}}
    timeout = config.get("API_TIMEOUT", 60) # Use configured timeout
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        dataset_id = r.json().get("data", {}).get("id")
        if dataset_id: print(f"      ✓ Dataset '{dataset_name}' created with ID: {dataset_id}"); return dataset_id
        else: print(f"      ✗ Dataset creation for '{dataset_name}' succeeded but no ID was returned."); return None
    except requests.RequestException as e:
        print(f"      ✗ Failed to create dataset '{dataset_name}': {e}"); return None

def post_fingerprint_via_api(config: Dict, token: str, org_id: str, dataset_id: str, fingerprint: Dict) -> bool:
    endpoint = config['CREATE_FINGERPRINT_ENDPOINT_TEMPLATE'].format(org_id=org_id, dataset_id=dataset_id)
    url = f"{config['API_BASE_URL']}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    api_payload = fingerprint
    
    # --- MODIFIED: Read the timeout from the config, with a safe default of 60 seconds ---
    timeout_seconds = config.get("API_TIMEOUT", 60)
    
    try:
        json_string_payload = json.dumps(api_payload, cls=NpEncoder)
    except TypeError as e:
        print(f"      ✗ ERROR: Failed to serialize fingerprint to JSON. {e}"); return False

    try:
        # --- MODIFIED: Use the configured timeout in the request ---
        r = requests.post(url, headers=headers, data=json_string_payload, timeout=timeout_seconds)
        r.raise_for_status()
        print(f"      ✓ POST successful. Response: {r.status_code}")
        return True
    except requests.RequestException as e:
        status = e.response.status_code if hasattr(e, "response") and e.response else "?"
        error_body = e.response.text if hasattr(e, "response") and e.response else "No response body."
        print(f"      ✗ POST failed ({status}): {e}\n        Response Body: {error_body}")
        return False