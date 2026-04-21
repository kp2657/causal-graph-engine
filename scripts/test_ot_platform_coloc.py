
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_query(name, query, variables):
    print(f"\nTesting {name}...")
    resp = httpx.post(OT_PLATFORM_GQL, json={"query": query, "variables": variables})
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        res = resp.json()
        if "errors" in res:
            print(f"GraphQL Errors: {json.dumps(res['errors'], indent=2)}")
        else:
            print("Success!")
    else:
        print(f"Error: {resp.text}")

# Test colocalisation on credibleSets
q = """
    query CredSetColoc($studyIds: [String!]!) {
      credibleSets(studyIds: $studyIds) {
        rows {
          colocalisation(studyTypes: [eqtl]) {
            rows {
              h4
              qtlGeneId
            }
          }
        }
      }
    }
    """
test_query("colocalisation", q, {"studyIds": ["GCST003116"]})
