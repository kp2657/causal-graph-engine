
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

# Corrected Assoc Query: Use target(...) { associatedDiseases(...) }
assoc_query_v2 = """
query AssocByEnsembl($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 1}) {
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
# Note: associatedDiseases doesn't support efoIds filter directly in some versions,
# we may have to filter in Python or use the disease -> associatedTargets path correctly.

# Let's try the correct disease -> associatedTargets path. 
# It usually uses 'page' and 'size', but how to filter by gene?
# In v4, you usually query Target directly.

test_query(OT_PLATFORM_GQL, assoc_query_v2, {"ensemblId": "ENSG00000169174", "efoId": "EFO_0001645"})
