import torch
import torch.optim as optim
import json
from sklearn.metrics import roc_auc_score
from data.graph_builder import MAbGraphBuilder
import torch_geometric.transforms as T

from models.kg_encoder import KGEncoder
from models.ab_encoder import AntibodyEncoder
from models.contrastive_alignment import ContrastiveAlignmentModule
from models.repurposing_head import RepurposingHead

def train():
    # 1. Prepare Data
    print("Loading Graph Data...")
    builder = MAbGraphBuilder()
    graph, mab_mapping, target_mapping, indication_mapping = builder.build_graph()
    
    # Split edges for link prediction on ('mab', 'treats', 'indication')
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        edge_types=[('mab', 'treats', 'indication')], # Specific edge type for evaluation
        add_negative_train_samples=False
    )
    
    train_data, val_data, test_data = transform(graph)
    
    with open("data/mab_features.json", "r") as f:
        features = json.load(f)
    
    idx_to_mab = {v: k for k, v in mab_mapping.items()}
    sequences = [features[idx_to_mab[i]]['heavy_chain'] for i in range(len(idx_to_mab))]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=====================================")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"=====================================\n")
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    print("Initializing Models...")
    hidden_dim = 64
    out_channels = 128
    
    kg_encoder = KGEncoder(
        hidden_channels=hidden_dim, 
        out_channels=out_channels,  
        num_layers=2, 
        node_types=graph.node_types, 
        metadata=graph.metadata()
    ).to(device)
    
    ab_encoder = AntibodyEncoder(out_channels=out_channels).to(device)
    contrastive_loss_fn = ContrastiveAlignmentModule().to(device)
    repurposing_head = RepurposingHead(in_channels=out_channels).to(device)
    
    optimizer = optim.Adam(
        list(kg_encoder.parameters()) + 
        list(ab_encoder.parameters()) + 
        list(repurposing_head.parameters()), 
        lr=5e-4, weight_decay=1e-5
    )

    import copy
    print("Starting Training Loop...")
    epochs = 300
    best_val_auc = 0.0
    best_models = None
    
    for epoch in range(epochs):
        kg_encoder.train()
        ab_encoder.train()
        repurposing_head.train()
        optimizer.zero_grad()
        
        # We need to compute embeddings using train_data.x_dict and train_data.edge_index_dict
        z_kg_dict = kg_encoder(train_data.x_dict, train_data.edge_index_dict)
        z_mab_kg = z_kg_dict['mab']
        z_indication_kg = z_kg_dict['indication']
        
        z_mab_seq = ab_encoder(sequences)
        loss_con = contrastive_loss_fn(z_mab_seq, z_mab_kg)
        
        edge_index_treats = train_data['mab', 'treats', 'indication'].edge_index
        if edge_index_treats.size(1) == 0:
            loss_link = torch.tensor(0.0, requires_grad=True, device=device)
        else:
            pos_src = edge_index_treats[0]
            pos_dst = edge_index_treats[1]
            pos_scores = repurposing_head(z_mab_kg[pos_src], z_indication_kg[pos_dst])
            
            neg_src = pos_src
            neg_dst = torch.randint(0, len(indication_mapping), pos_dst.size(), device=device)
            neg_scores = repurposing_head(z_mab_kg[neg_src], z_indication_kg[neg_dst])
            
            loss_link = repurposing_head.link_prediction_loss(pos_scores, neg_scores)
        
        total_loss = loss_con + loss_link
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Eval on val set
            kg_encoder.eval()
            ab_encoder.eval()
            repurposing_head.eval()
            with torch.no_grad():
                z_val_kg_dict = kg_encoder(val_data.x_dict, val_data.edge_index_dict)
                z_val_mab_kg = z_val_kg_dict['mab']
                z_val_ind_kg = z_val_kg_dict['indication']
                
                val_edge_label_index = val_data['mab', 'treats', 'indication'].edge_label_index
                val_edge_label = val_data['mab', 'treats', 'indication'].edge_label
                if val_edge_label_index.size(1) > 0:
                    src, dst = val_edge_label_index[0], val_edge_label_index[1]
                    scores = repurposing_head(z_val_mab_kg[src], z_val_ind_kg[dst]).cpu().numpy()
                    labels = val_edge_label.cpu().numpy()
                    
                    try:
                        val_auc = roc_auc_score(labels, scores)
                        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | Val AUROC: {val_auc:.4f}")
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_models = (
                                copy.deepcopy(kg_encoder.state_dict()),
                                copy.deepcopy(ab_encoder.state_dict()),
                                copy.deepcopy(repurposing_head.state_dict())
                            )
                    except ValueError:
                        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f}")

    print("\n--- Evaluation on Test Set ---")
    if best_models is not None:
        kg_encoder.load_state_dict(best_models[0])
        ab_encoder.load_state_dict(best_models[1])
        repurposing_head.load_state_dict(best_models[2])
    
    kg_encoder.eval()
    ab_encoder.eval()
    repurposing_head.eval()
    
    with torch.no_grad():
        # Using test_data for evaluation
        z_kg_dict = kg_encoder(test_data.x_dict, test_data.edge_index_dict)
        z_mab_kg = z_kg_dict['mab']
        z_indication_kg = z_kg_dict['indication']
        
        edge_label_index = test_data['mab', 'treats', 'indication'].edge_label_index
        edge_label = test_data['mab', 'treats', 'indication'].edge_label
        
        src, dst = edge_label_index[0], edge_label_index[1]
        scores = repurposing_head(z_mab_kg[src], z_indication_kg[dst]).cpu().numpy()
        labels = edge_label.cpu().numpy()
        
        try:
            auc = roc_auc_score(labels, scores)
            print(f"Test AUROC for Repurposing Head: {auc:.4f}")
        except ValueError:
            print("Only one class present in test labels. AUROC cannot be computed.")

if __name__ == "__main__":
    train()
