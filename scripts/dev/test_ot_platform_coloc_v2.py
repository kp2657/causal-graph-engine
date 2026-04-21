
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_query(name, query, variables):
    print(f"\nTesting {name}...")
    resp = httpx.post(OT_PLATFORM_GQL, json={"query": query, "variables": variables})
    if resp.status_code == 200:
        res = resp.json()
        if "errors" in res:
            print(f"GraphQL Errors: {json.dumps(res['errors'], indent=2)}")
        else:
            print("Success!")
            # print(json.dumps(res['data'], indent=2))
    else:
        print(f"Status: {resp.status_code}, Error: {resp.text}")

# Test colocalisation variations
q = """
    query CredSetColoc($studyIds: [String!]!) {
      credibleSets(studyIds: $studyIds) {
        rows {
          colocalisation(studyTypes: [eqtl]) {
            rows {
              h4
              otherStudyLocus {
                studyId
                qtlGeneId
              }
            }
          }
        }
      }
    }
    """
test_query("colocalisation_v2", q, {"studyIds": ["GCST003116"]})
