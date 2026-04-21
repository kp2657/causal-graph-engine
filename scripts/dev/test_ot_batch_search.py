
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_batch_search(symbols):
    # Try space-separated query string
    q_str = " ".join(symbols)
    print(f"\nTesting Batch Search for: {symbols}")
    
    query = """
    query SearchTargets($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 50}) {
        hits { id name }
      }
    }
    """
    resp = httpx.post(OT_PLATFORM_GQL, json={"query": query, "variables": {"q": q_str}})
    if resp.status_code == 200:
        hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
        print(f"Found {len(hits)} hits.")
        for h in hits:
            print(f"  {h['name']} -> {h['id']}")
    else:
        print(f"Status: {resp.status_code}, Error: {resp.text}")

test_batch_search(["PCSK9", "LDLR", "HMGCR"])
