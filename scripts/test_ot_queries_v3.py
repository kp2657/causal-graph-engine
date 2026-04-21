
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_query(url, query, variables):
    print(f"\nTesting query at {url}...")
    resp = httpx.post(url, json={"query": query, "variables": variables})
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
    else:
        res = resp.json()
        if "errors" in res:
            print(f"GraphQL Errors: {json.dumps(res['errors'], indent=2)}")
        else:
            print("Success!")
            # print(json.dumps(res['data'], indent=2))

# V3: Use target(...) { associations(diseaseIds: ...) } 
# This is the correct v4 way to get specific target-disease scores.
assoc_query_v3 = """
query AssocByEnsembl($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 10}) {
      rows {
        disease {
          id
        }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""
test_query(OT_PLATFORM_GQL, assoc_query_v3, {"ensemblId": "ENSG00000169174", "efoId": "EFO_0001645"})
