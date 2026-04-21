
import httpx
import json

URL = "https://www.ebi.ac.uk/eqtl/api/v2/datasets"

def test_eqtl_api(params):
    print(f"\nTesting eQTL API with params: {params}")
    resp = httpx.get(URL, params=params)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", [])
        print(f"Found {len(results)} results.")
        if results:
            first = results[0]
            print(f"First result: study_id={first.get('study_id')}, study_label={first.get('study_label')}")
    else:
        print(f"Error: {resp.text}")

# Try common study labels/ids
test_eqtl_api({"quant_method": "ge", "size": 5, "study_id": "OneK1K"})
test_eqtl_api({"quant_method": "ge", "size": 5, "study_label": "OneK1K"})
test_eqtl_api({"quant_method": "ge", "size": 5, "study_label": "BLUEPRINT"})
test_eqtl_api({"quant_method": "ge", "size": 5, "study_label": "Yazar2022"})
