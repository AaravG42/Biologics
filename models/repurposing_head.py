import torch
import torch.nn as nn

class RepurposingHead(nn.Module):
    def __init__(self, in_channels):
        """
        Decoder layer optimizing margin ranking or logistic loss on (mAb, treats, cancer_type) edges.
        
        Args:
            in_channels (int): Embedding dimensions (should match out_channels of encoders).
        """
        super().__init__()
        # Simple bilinear decoder for (mAb, relation, indication)
        # Using a parameterized relation embedding for 'treats'
        self.relation_embed = nn.Parameter(torch.Tensor(in_channels))
        nn.init.xavier_uniform_(self.relation_embed.unsqueeze(0))

    def forward(self, z_mab, z_indication):
        """
        Score the likelihood of a link between a set of mAbs and indications.
        (z_mab * R) dot z_indication
        
        Args:
            z_mab (Tensor): [batch_size, in_channels]
            z_indication (Tensor): [batch_size, in_channels]
            
        Returns:
            scores (Tensor): [batch_size] link scores.
        """
        # Element-wise multiplication with relation embedding
        mab_rel = z_mab * self.relation_embed
        
        # Dot product with target indication embedding
        scores = torch.sum(mab_rel * z_indication, dim=-1)
        return scores
        
    def link_prediction_loss(self, pos_scores, neg_scores, margin=1.0):
        """
        Computes Margin Ranking Loss.
        """
        y = torch.ones_like(pos_scores)
        loss_fn = nn.MarginRankingLoss(margin=margin)
        return loss_fn(pos_scores, neg_scores, y)

if __name__ == "__main__":
    head = RepurposingHead(128)
    z_m = torch.randn(4, 128)
    z_i = torch.randn(4, 128)
    scores = head(z_m, z_i)
    print("Scores:", scores)
