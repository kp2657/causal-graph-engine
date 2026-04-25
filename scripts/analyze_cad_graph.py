import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(os.getcwd()) / "causal-graph-engine"))

from graph.db import GraphDB

def run_analysis():
    db_path = "./causal-graph-engine/data/graph.kuzu"
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        return

    db = GraphDB(db_path)
    
    # Debug: Check node/rel counts
    res = db._conn.execute("MATCH (g:Gene) RETURN count(g)")
    print(f"Total Genes in DB: {res.get_next()[0]}")
    res = db._conn.execute("MATCH (p:CellularProgram) RETURN count(p)")
    print(f"Total Programs in DB: {res.get_next()[0]}")
    res = db._conn.execute("MATCH ()-[r:RegulatesProgram]->() RETURN count(r)")
    print(f"Total RegulatesProgram edges: {res.get_next()[0]}")

    target_trait = 'coronary artery disease'

    print(f"\n=== 1. Master Switch Analysis ===")
    query_master = f"""
    MATCH (g:Gene)-[:RegulatesProgram]->(p:CellularProgram)-[:DrivesTrait]->(t:DiseaseTrait {{id: '{target_trait}'}})
    RETURN g.id, count(p) as program_count
    ORDER BY program_count DESC
    LIMIT 10
    """
    result = db._conn.execute(query_master)
    while result.has_next():
        row = result.get_next()
        print(f"Gene: {row[0]:<15} | Regulates {row[1]} Programs")

if __name__ == "__main__":
    run_analysis()
