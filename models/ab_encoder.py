import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()

class AntibodyEncoder(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", out_channels=128):
        """
        Antibody Sequence Encoder wrapping ESM-2.
        
        Args:
            model_name (str): HuggingFace hub model name for ESM-2.
                              Using 8M model for lightweight prototyping.
            out_channels (int): The shared latent space dimension (size of z_seq).
        """
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the base model without the LM head
        self.esm_model = AutoModel.from_pretrained(model_name)
        
        # We'll extract raw hidden states, not logits.
        self.hidden_size = self.esm_model.config.hidden_size
        
        # Projection layer to map to shared contrastive space
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, out_channels)
        )

    def forward(self, sequences):
        """
        Args:
            sequences (list of str): A list of antibody sequences (heavy and light chains concatenated).
            
        Returns:
            z_seq (Tensor): [batch_size, out_channels] sequence embeddings ready for contrastive alignment.
        """
        device = next(self.parameters()).device
        
        # Tokenize
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass through ESM
        # AutoModelForMaskedLM returns MaskedLMOutput, we need to get hidden states.
        # ESM models usually return hidden_states if requested.
        outputs = self.esm_model(**inputs, output_hidden_states=True)
        
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Mean pooling over the sequence length, ignoring padding tokens
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Pass through projector
        z_seq = self.projector(pooled_output)
        
        # L2 Normalize the embeddings for InfoNCE loss
        z_seq = nn.functional.normalize(z_seq, dim=1)
        
        return z_seq

if __name__ == "__main__":
    encoder = AntibodyEncoder()
    mock_seqs = ["QVQLVQSGVEVKKPGASVKVSCKAS", "EVQLVESGGGLVQPGGSLRLSCAAS"]
    embeddings = encoder(mock_seqs)
    print("Encoded Shape:", embeddings.shape)
