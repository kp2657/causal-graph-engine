
import httpx
import json

URL = "https://www.ebi.ac.uk/eqtl/api/v2/datasets"

def test_eqtl_api(params):
    print(f"\nTesting eQTL API with params: {params}")
    resp = httpx.get(URL, params=params)
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
    else:
        print("Success!")
        # print(json.dumps(resp.json(), indent=2))

# The failing parameters from the log
test_eqtl_api({
    "quant_method": "ge",
    "size": 50,
    "start": 0,
    "study_label": "Yazar2022"
})

# Try without study_label
test_eqtl_api({
    "quant_method": "ge",
    "size": 10,
    "start": 0
})
