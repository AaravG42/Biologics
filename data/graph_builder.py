import pandas as pd
import json
import torch
from torch_geometric.data import HeteroData

class MAbGraphBuilder:
    def __init__(self, kg_path="data/mab_kg.csv", features_path="data/mab_features.json"):
        self.kg_path = kg_path
        self.features_path = features_path
        
    def build_graph(self):
        # Load data
        kg_df = pd.read_csv(self.kg_path)
        with open(self.features_path, 'r') as f:
            features = json.load(f)
            
        # Extract unique entities
        mabs = list(features.keys())
        targets = list(kg_df['Target'].unique())
        indications = list(kg_df['Indication'].unique())
        
        # Create mapping from name to integer ID
        mab_mapping = {name: i for i, name in enumerate(mabs)}
        target_mapping = {name: i for i, name in enumerate(targets)}
        indication_mapping = {name: i for i, name in enumerate(indications)}
        
        # Initialize PyG HeteroData
        data = HeteroData()
        
        # Add basic dummy features for nodes (to be replaced by ESM/embedding encoders later)
        # Using a default hidden dimension of 64
        hidden_dim = 64
        data['mab'].x = torch.randn(len(mabs), hidden_dim)
        data['target'].x = torch.randn(len(targets), hidden_dim)
        data['indication'].x = torch.randn(len(indications), hidden_dim)
        
        # Build Edges
        mab_target_src = []
        mab_target_dst = []
        mab_indication_src = []
        mab_indication_dst = []
        
        for idx, row in kg_df.iterrows():
            mab_idx = mab_mapping[row['mAb']]
            if pd.notna(row['Target']):
                target_idx = target_mapping[row['Target']]
                mab_target_src.append(mab_idx)
                mab_target_dst.append(target_idx)
                
            if pd.notna(row['Indication']):
                ind_idx = indication_mapping[row['Indication']]
                mab_indication_src.append(mab_idx)
                mab_indication_dst.append(ind_idx)
                
        # Add edge indices to graph
        data['mab', 'targets', 'target'].edge_index = torch.tensor([mab_target_src, mab_target_dst], dtype=torch.long)
        data['mab', 'treats', 'indication'].edge_index = torch.tensor([mab_indication_src, mab_indication_dst], dtype=torch.long)
        
        # Print basic graph info
        print("Graph Construction Complete:")
        print(data)
        return data, mab_mapping, target_mapping, indication_mapping

if __name__ == "__main__":
    builder = MAbGraphBuilder()
    graph, mmap, tmap, imap = builder.build_graph()
