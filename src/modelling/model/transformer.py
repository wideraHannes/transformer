from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from config.paths import MODEL_CONFIG
from src.modelling.embedding.transformer_embedding import TransformerEmbedding
from src.modelling.model.decoder import Decoder
from src.modelling.model.encoder import Encoder


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""

    vocab_size: int
    d_model: int
    n_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    max_len: int

    def validate(self):
        """Validates the configuration to ensure all fields are properly set."""
        if not (self.vocab_size > 0):
            raise ValueError("vocab_size must be a positive integer.")
        if not (self.d_model > 0):
            raise ValueError("d_model must be a positive integer.")
        if not (self.n_heads > 0):
            raise ValueError("n_heads must be a positive integer.")
        if not (self.num_encoder_layers > 0):
            raise ValueError("n_layers must be a positive integer.")
        if not (self.num_decoder_layers > 0):
            raise ValueError("n_layers must be a positive integer.")
        if not (self.dim_feedforward > 0):
            raise ValueError("dim_feedforward must be a positive integer.")
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1.")
        if not (self.max_len > 0):
            raise ValueError("max_len must be a positive integer.")


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerModel, self).__init__()

        # Validate configuration
        config.validate()

        # Assign configurations
        self.config = config

        self.src_embed = TransformerEmbedding(
            config.vocab_size, config.d_model, config.max_len
        )
        self.tgt_embed = TransformerEmbedding(
            config.vocab_size, config.d_model, config.max_len
        )

        # Initialize encoder and decoder
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            n_layers=config.num_encoder_layers,
        )
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            n_layers=config.num_decoder_layers,
        )
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self):
        # Share the token embedding weights between src_embed and tgt_embed
        self.src_embed.token_emb.weight = self.tgt_embed.token_emb.weight
        # Share the token embedding weights with the output layer
        self.output_layer.weight = self.tgt_embed.token_emb.weight

    def generate(self, src, max_length=50):
        src_emb = self.src_embed(src)
        memory = self.encoder(src_emb)
        outputs = []
        tgt = torch.tensor(
            [[self.eos_token_id]], device=src.device
        )  # Start with <EOS> token
        for _ in range(max_length):
            tgt_emb = self.embedding(tgt)
            output = self.decoder(tgt_emb, memory)
            logits = self.output_projection(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # Greedy decoding
            tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=-1)
            outputs.append(next_token)
            if next_token.item() == self.eos_token_id:  # Stop if EOS token is generated
                break
        return tgt

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.src_embed(src)
        encoder_out = self.encoder(src_emb, src_mask)
        tgt_emb = self.tgt_embed(tgt)
        decoder_out = self.decoder(encoder_out, tgt_emb, src_mask, tgt_mask)
        output = self.output_layer(decoder_out)
        return output


if __name__ == "__main__":
    with open(MODEL_CONFIG, "r") as file:
        transformer_config = TransformerConfig(**yaml.safe_load(file))

    # Define test data for hidden states and attention masks
    src = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    tgt = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    src_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    tgt_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    memory_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])

    # Initialize the Transformer model
    model = TransformerModel(config=transformer_config)

    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)

    probs = F.softmax(output, dim=-1)

    # Get the most likely token indices
    token_ids = torch.argmax(probs, dim=-1)

    # Convert token IDs to tokens
    print("Token IDs:", token_ids)
