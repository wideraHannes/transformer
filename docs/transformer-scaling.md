| Dataset Size | d_model | n_heads | encoder/decoder layers | dim_feedforward | Batch Size | Total Params* | Notes |
|--------------|---------|----------|----------------------|-----------------|------------|---------------|--------|
| 5,000        | 128     | 2        | 2/2                  | 512            | 64         | ~2M          | Minimal viable setup |
| 10,000       | 256     | 4        | 4/4                  | 1024           | 128        | ~8M          | Good starting point |
| 25,000       | 384     | 6        | 4/4                  | 1536           | 128        | ~18M         | Initial validation |
| 50,000       | 512     | 8        | 6/6                  | 2048           | 256        | ~43M         | Quarter-scale transformer |
| 100,000      | 768     | 8        | 6/6                  | 3072           | 256        | ~89M         | Half-scale transformer |
| 250,000      | 1024    | 16       | 6/6                  | 4096           | 512        | ~210M        | Approaching base size |
| 500,000      | 512     | 8        | 6/6                  | 2048           | 4096*      | ~43M         | Original base settings |
| 1,000,000    | 512     | 8        | 6/6                  | 2048           | 4096*      | ~43M         | Original base settings |
| 2,000,000    | 512     | 8        | 6/6                  | 2048           | 4096*      | ~43M         | Original base settings |
| 4,500,000    | 512     | 8        | 6/6                  | 2048           | 4096*      | ~43M         | Original paper config |

*The original transformer used tokens per batch instead of batch size, with 25,000 tokens per batch accumulated

Original paper ("base" model) settings:
- d_model: 512
- n_heads: 8
- encoder/decoder layers: 6/6
- dim_feedforward: 2048
- dropout: 0.1
- Learning rate: Custom warmup schedule
- Label smoothing: 0.1
- Max position embeddings: 512
- vocab_size: 37,000 (BPE)

Key differences in the original:
1. Used much larger effective batch sizes (25,000 tokens per batch)
2. Used more sophisticated regularization
3. Trained for 100,000 steps with custom LR schedule
4. Used custom BPE tokenization
5. Accumulated gradients for larger effective batches

Training recommendations per size:
- 5K-25K: 20-30 epochs
- 25K-100K: 15-20 epochs
- 100K-500K: 10-15 epochs
- 500K+: Follow original paper's step count (~100K steps)
