In training, the model is typically trained using a method called "teacher forcing," where the ground truth tokens are fed into the decoder at each time step. This allows the model to learn the correct sequence of tokens more efficiently. During training, the model learns to predict the next token in the sequence given the previous tokens and the source sentence.

In contrast, during inference (or evaluation), we do not have access to the ground truth tokens. Instead, we need to generate the sequence token by token. This is where autoregressive generation comes into play. In autoregressive generation, the model generates one token at a time and uses the previously generated tokens as input for generating the next token. This process continues until an end-of-sequence (EOS) token is generated or a maximum length is reached.

Key Differences:
Training (Teacher Forcing):

Ground truth tokens are used as input to the decoder at each time step.
The model learns to predict the next token in the sequence given the previous tokens and the source sentence.
This method helps the model learn the correct sequence more efficiently.
Inference (Autoregressive Generation):

The model generates one token at a time.
The previously generated tokens are used as input for generating the next token.
This process continues until an EOS token is generated or a maximum length is reached.
Greedy decoding is one method of autoregressive generation where the token with the highest probability is selected at each step.
