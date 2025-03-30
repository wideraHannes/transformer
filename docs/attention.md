### **Attention Module Overview**

The `Attention` module implements the **Scaled Dot-Product Attention** mechanism and supports both self-attention and cross-attention. This module can handle variable input sizes and incorporates two types of masking: **future masking** (to prevent information leakage in autoregressive models) and **custom attention masking** (to account for padding or other restrictions). Below is a concise breakdown of its functionality.

---

#### **Input Dimensions**

1. **Query (q)**: Shape \((B, L_q, d_q)\)
   - \(B\): Batch size
   - \(L_q\): Query sequence length
   - \(d_q\): Query feature dimension
2. **Key (k)**: Shape \((B, L_k, d_k)\)

   - \(L_k\): Key sequence length
   - \(d_k\): Key feature dimension

3. **Value (v)**: Shape \((B, L_k, d_v)\)

   - \(d_v\): Value feature dimension

4. **Mask**: Optional, shape can vary depending on the context:
   - For self-attention, \((B, L_q)\) or \((B, L_q, L_k)\)
   - For cross-attention, \((B, L_k)\) or \((B, 1, L_k)\)

---

#### **Steps**

1. **Scaled Dot-Product Attention**:

   - Compute attention scores:
     \[
     \text{scores} = \frac{q \cdot k^T}{\sqrt{d_k}}
     \]
     Resulting shape: \((B, L_q, L_k)\).

2. **Masking**:

   - **Custom Mask**: Applied to ensure attention scores respect padding or other constraints.
     - If \( \text{mask} \) is provided, positions corresponding to `0` are replaced with \(-\infty\).
   - **Future Masking** (optional): Prevents attention to future positions in autoregressive tasks.
     - Implements an upper triangular mask to block future tokens.

3. **Softmax**:

   - Apply softmax to the masked scores along the last dimension (\(L_k\)), producing attention weights of shape \((B, L_q, L_k)\).

4. **Weighted Sum**:
   - Compute the weighted sum of the values:
     \[
     \text{output} = \text{attn_weights} \cdot v
     \]
     Resulting shape: \((B, L_q, d_v)\).

---

#### **Output Dimensions**

- The final output has the shape \((B, L_q, d_v)\), where \(L_q\) matches the query sequence length, and \(d_v\) matches the value feature dimension.

---

#### **Use Cases**

1. **Self-Attention**:

   - Queries, keys, and values are identical (\(q = k = v\)).
   - Mask typically blocks padded tokens or irrelevant positions.

2. **Future-Masked Self-Attention**:

   - Prevents queries from attending to future positions using a triangular mask.
   - Useful for tasks like language modeling or sequence generation.

3. **Cross-Attention**:
   - Queries come from one sequence, while keys and values come from another.
   - Handles \(L_q \neq L_k\), allowing attention across sequences of different lengths.
   - Mask restricts attention to valid positions in the key/value sequence.

---

#### **Key Highlights**

- **Masking Flexibility**: Supports custom and future masks simultaneously, ensuring proper attention application for a wide range of tasks.
- **Generalized Dot-Product**: Handles mismatched sequence lengths (\(L_q \neq L_k\)) for cross-attention scenarios.
- **Output Shape Consistency**: Guarantees output shape is always \((B, L_q, d_v)\), regardless of input dimensions.

This design makes the `Attention` module versatile and robust for self-attention and cross-attention in both encoder-decoder and autoregressive architectures.
