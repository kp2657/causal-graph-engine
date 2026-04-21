
import httpx
import json

# The URL that failed in the log
URL = "https://www.ebi.ac.uk/eqtl/api/v2/datasets/QTD000609/associations"
params = {"gene_id": "ENSG00000000971", "size": 20}

print(f"Testing eQTL Association API: {URL} with params {params}")
resp = httpx.get(URL, params=params)
print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error Body: {resp.text}")
else:
    print("Success!")
