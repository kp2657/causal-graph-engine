
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

# 1. Test 'studies' entry point on Platform API
q1 = """
    query Studies($efoId: String!) {
      studies(diseaseIds: [$efoId]) {
        rows { id studyType }
      }
    }
    """
test_query("studies", q1, {"efoId": "EFO_0001645"})

# 2. Test 'credibleSets' entry point on Platform API
q2 = """
    query CredSets($studyIds: [String!]!) {
      credibleSets(studyIds: $studyIds) {
        rows { studyLocusId }
      }
    }
    """
test_query("credibleSets", q2, {"studyIds": ["GCST003116"]})
