import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveAlignmentModule(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Symmetric InfoNCE Loss for aligning sequence/structure embeddings (z_seq)
        with Knowledge Graph embeddings (z_KG).
        
        Args:
            temperature (float): Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_seq, z_kg):
        """
        Compute symmetric InfoNCE loss between a batch of sequences and corresponding KG nodes.
        Assumes z_seq and z_kg are already L2 normalized and paired row-by-row.
        
        Args:
            z_seq (Tensor): [batch_size, out_channels] sequence embeddings.
            z_kg (Tensor): [batch_size, out_channels] node embeddings.
            
        Returns:
            loss (Tensor): The contrastive loss scalar.
        """
        # Ensure we have normalized embeddings
        z_seq_norm = F.normalize(z_seq, p=2, dim=1)
        z_kg_norm = F.normalize(z_kg, p=2, dim=1)
        
        # Cosine similarity matrix
        # [batch_size, batch_size]
        logits = torch.matmul(z_seq_norm, z_kg_norm.t()) / self.temperature
        
        # Labels are the diagonal (cognate pairs)
        batch_size = z_seq.shape[0]
        labels = torch.arange(batch_size, device=z_seq.device)
        
        # Compute loss in both directions (seq -> kg, and kg -> seq)
        loss_seq_to_kg = F.cross_entropy(logits, labels)
        loss_kg_to_seq = F.cross_entropy(logits.t(), labels)
        
        # Total symmetric loss
        loss = (loss_seq_to_kg + loss_kg_to_seq) / 2.0
        return loss

if __name__ == "__main__":
    module = ContrastiveAlignmentModule()
    z_s_mock = torch.randn(4, 128)
    z_k_mock = torch.randn(4, 128)
    loss = module(z_s_mock, z_k_mock)
    print("Contrastive Loss:", loss.item())
