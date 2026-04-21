
import httpx
import json

OT_GENETICS_GQL = "https://api.genetics.opentargets.org/graphql"

def test_query(name, query, variables):
    print(f"\nTesting {name} at {OT_GENETICS_GQL}...")
    resp = httpx.post(OT_GENETICS_GQL, json={"query": query, "variables": variables})
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
    else:
        res = resp.json()
        if "errors" in res:
            print(f"GraphQL Errors: {json.dumps(res['errors'], indent=2)}")
        else:
            print("Success!")

# Query: Studies
q = """
    query Studies($efoId: String!) {
      studiesAndLeadVariantsForDisease(efoId: $efoId) {
        study {
          studyId
        }
      }
    }
    """
test_query("studiesAndLeadVariantsForDisease", q, {"efoId": "EFO_0001645"})
