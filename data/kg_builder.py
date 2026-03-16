import os
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import json

class KGBuilder:
    def __init__(self, endpoint_url="http://www.imgt.org/mAb-KG/sparql"):
        self.sparql = SPARQLWrapper(endpoint_url)
        # Using a default JSON return format
        self.sparql.setReturnFormat(JSON)

    def fetch_mab_targets(self, limit=100):
        """
        Fetch mAb to Target relationships.
        Placeholder SPARQL query - adjust based on exact IMGT/mAb-KG ontology.
        """
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX imgt: <http://www.imgt.org/ontology/>
        
        SELECT ?mab ?target ?indication
        WHERE {{
            ?mab rdf:type imgt:MonoclonalAntibody .
            OPTIONAL {{ ?mab imgt:targets ?target . }}
            OPTIONAL {{ ?mab imgt:treats ?indication . }}
        }}
        LIMIT {limit}
        """
        self.sparql.setQuery(query)
        
        try:
            results = self.sparql.query().convert()
            data = []
            for result in results["results"]["bindings"]:
                mab = result.get("mab", {}).get("value", "Unknown")
                target = result.get("target", {}).get("value", "Unknown")
                indication = result.get("indication", {}).get("value", "Unknown")
                data.append({"mAb": mab, "Target": target, "Indication": indication})
            
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error querying SPARQL endpoint: {e}")
            # If endpoint is unreachable or query is malformed, return dummy data for development
            print("Returning dummy data for testing...")
            return pd.DataFrame([
                {"mAb": "Pembrolizumab", "Target": "PD-1", "Indication": "Melanoma"},
                {"mAb": "Trastuzumab", "Target": "HER2", "Indication": "Breast Cancer"},
                {"mAb": "Rituximab", "Target": "CD20", "Indication": "Lymphoma"}
            ])

    def save_graph(self, df, output_path="data/mab_kg.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved KG data to {output_path}")

if __name__ == "__main__":
    builder = KGBuilder()
    df = builder.fetch_mab_targets()
    print(df.head())
    builder.save_graph(df)
