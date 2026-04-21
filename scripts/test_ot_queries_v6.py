
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
    else:
        print(f"Status: {resp.status_code}, Error: {resp.text}")

# V6: Try disease -> associatedTargets(ensemblIds:) WITH evidences
# Note: Earlier 400 was on 'associatedTargets', but maybe it works if structured like this:
q = """
query DiseaseGeneEvidences($efoId: String!, $ensemblId: String!) {
  disease(efoId: $efoId) {
    associatedTargets(ensemblIds: [$ensemblId]) {
      rows {
        target { id approvedSymbol }
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
test_query("DiseaseGeneEvidences", q, {"efoId": "EFO_0001645", "ensemblId": "ENSG00000169174"})
