
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_GENETICS_GQL = "https://api.genetics.opentargets.org/graphql"

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

# 1. Test Search Query (used in multiple places)
search_query = """
query SearchTarget($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits {
      id
      name
    }
  }
}
"""
test_query(OT_PLATFORM_GQL, search_query, {"q": "PCSK9"})

# 2. Test Assoc Query (used in _get_ot_genetic_scores_live)
assoc_query = """
query AssocByEnsembl($efoId: String!, $ensemblIds: [String!]!) {
  disease(efoId: $efoId) {
    associatedTargets(ensemblIds: $ensemblIds, page: {index: 0, size: 10}) {
      rows {
        target {
          approvedSymbol
        }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""
test_query(OT_PLATFORM_GQL, assoc_query, {"efoId": "EFO_0001645", "ensemblIds": ["ENSG00000169174"]})

# 3. Test CredSetEvidence Query (used in get_ot_genetic_instruments)
evid_q = """
query CredSetEvidence($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    evidences(
      efoIds: [$efoId]
      datasourceIds: ["gwas_credible_sets"]
      size: 5
    ) {
      rows {
        score
      }
    }
  }
}
"""
test_query(OT_PLATFORM_GQL, evid_q, {"ensemblId": "ENSG00000169174", "efoId": "EFO_0001645"})
