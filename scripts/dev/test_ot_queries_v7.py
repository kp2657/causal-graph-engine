
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

# V7: Use target -> associatedDiseases -> evidences
q = """
query DiseaseGeneEvidences($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    associatedDiseases {
      rows {
        disease { id }
        evidences(datasourceIds: ["gwas_credible_sets"]) {
          rows {
            score
            credibleSet {
              beta
              standardError
              pValueMantissa
              pValueExponent
              studyType
              studyId
            }
          }
        }
      }
    }
  }
}
"""
test_query("DiseaseGeneEvidences_v7", q, {"ensemblId": "ENSG00000169174", "efoId": "EFO_0001645"})
