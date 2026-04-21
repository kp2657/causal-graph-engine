
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_query(name, query, variables):
    print(f"\nTesting {name}...")
    resp = httpx.post(OT_PLATFORM_GQL, json={"query": query, "variables": variables})
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
    else:
        res = resp.json()
        if "errors" in res:
            print(f"GraphQL Errors: {json.dumps(res['errors'], indent=2)}")
        else:
            print("Success!")

# Query: DrugSearch
q = """
    query DrugSearch($term: String!) {
      search(queryString: $term, entityNames: ["drug"]) {
        hits {
          id
          entity
          name
          object {
            ... on Drug {
              id
              name
              maximumClinicalStage
              mechanismsOfAction {
                rows {
                  actionType
                  targets {
                    approvedSymbol
                  }
                }
              }
            }
          }
        }
      }
    }
    """
test_query("DrugSearch", q, {"term": "atorvastatin"})
