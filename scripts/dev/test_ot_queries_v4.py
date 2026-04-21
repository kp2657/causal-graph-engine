
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

# Query 1: DiseaseTargets (get_open_targets_disease_targets)
q1 = """
    query DiseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
          count
          rows {
            target {
              id
              approvedSymbol
              tractability {
                label
                modality
                value
              }
              drugAndClinicalCandidates {
                count
                rows {
                  drug {
                    id
                    name
                  }
                  maxClinicalStage
                }
              }
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
test_query("DiseaseTargets", q1, {"efoId": "EFO_0001645", "size": 2})

# Query 2: TargetInfo (get_open_targets_target_info)
q2 = """
    query TargetInfo($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        tractability {
          label
          modality
          value
        }
        drugAndClinicalCandidates {
          count
          rows {
            drug {
              id
              name
            }
            maxClinicalStage
          }
        }
      }
    }
    """
test_query("TargetInfo", q2, {"ensemblId": "ENSG00000169174"})
