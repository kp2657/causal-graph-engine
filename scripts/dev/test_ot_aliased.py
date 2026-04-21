
import httpx
import json

OT_PLATFORM_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

def test_aliased_queries(symbols):
    # Construct an aliased query
    # {
    #   g0: search(queryString: "PCSK9", entityNames: ["target"], page: {size: 1}) { hits { id name } }
    #   g1: search(queryString: "LDLR", entityNames: ["target"], page: {size: 1}) { hits { id name } }
    # }
    
    aliased_parts = []
    for i, sym in enumerate(symbols):
        aliased_parts.append(
            f'g{i}: search(queryString: "{sym}", entityNames: ["target"], page: {{index: 0, size: 1}}) {{ hits {{ id name }} }}'
        )
    
    query = "{ " + " ".join(aliased_parts) + " }"
    print(f"\nTesting Aliased Query for: {symbols}")
    
    resp = httpx.post(OT_PLATFORM_GQL, json={"query": query})
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        print(f"Found {len(data)} results.")
        for alias, result in data.items():
            hits = result.get("hits", [])
            if hits:
                print(f"  {alias} -> {hits[0]['name']} ({hits[0]['id']})")
    else:
        print(f"Status: {resp.status_code}, Error: {resp.text}")

test_aliased_queries(["PCSK9", "LDLR", "HMGCR", "NOD2"])
