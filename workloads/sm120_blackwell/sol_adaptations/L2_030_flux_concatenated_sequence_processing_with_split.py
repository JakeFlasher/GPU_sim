import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flux concatenated sequence processing pattern.
    
    1. Concatenate encoder_hidden_states and hidden_states along sequence dimension
    2. Apply linear projection to the concatenated sequence
    3. Split back into separate encoder and image streams
    
    Args:
        hidden_states: Image latent sequence [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: Text conditioning sequence [batch, text_seq_len, hidden_dim]
        process_weight: Linear projection weight [hidden_dim, hidden_dim]
        
    Returns:
        Tuple of (processed_encoder_hidden_states, processed_hidden_states)
    """
    text_seq_len = encoder_hidden_states.shape[1]
    img_seq_len = hidden_states.shape[1]
    
    # Step 1: Concatenate sequences along sequence dimension
    # Shape: [batch, text_seq_len + img_seq_len, hidden_dim]
    # Allocate and place the two streams explicitly; this preserves the
    # concatenation semantics without a large CatArrayBatchedCopy launch
    # under SM120 instruction instrumentation.
    concatenated = torch.empty((hidden_states.shape[0], text_seq_len + img_seq_len, hidden_states.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype)
    concatenated[:, :text_seq_len, :] = encoder_hidden_states
    concatenated[:, text_seq_len:, :] = hidden_states
    
    # Step 2: Apply linear projection (no bias)
    # processed = concatenated @ process_weight.T
    processed = torch.matmul(concatenated, process_weight.t())
    
    # Step 3: Split back into separate streams
    processed_encoder = processed[:, :text_seq_len, :]
    processed_hidden = processed[:, text_seq_len:, :]
    
    return processed_encoder, processed_hidden
