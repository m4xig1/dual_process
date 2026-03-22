"""P-tuning utilities: trainable soft prompt embeddings prepended to model inputs.

Provides ``SoftPromptEmbeddings``, a small trainable module whose soft tokens
are concatenated in front of the encoder hidden-states (for the Flux T5 text
encoder) or the VLM input embeddings before each forward pass.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SoftPromptEmbeddings(nn.Module):
    """Trainable soft prompt embeddings for P-tuning.

    Holds a set of ``num_tokens`` learnable vectors of dimension
    ``hidden_size``.  Call :meth:`prepend` to concatenate them in front of any
    ``[batch, seq_len, hidden_size]`` tensor before feeding it to a model.

    Args:
        num_tokens: Number of soft prompt tokens.
        hidden_size: Dimension of each token embedding.
        init_std: Standard deviation for Gaussian initialisation.
    """

    def __init__(self, num_tokens: int, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.soft_tokens = nn.Parameter(
            torch.randn(1, num_tokens, hidden_size) * init_std
        )

    def prepend(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prepend soft prompt tokens to *hidden_states*.

        Args:
            hidden_states: ``[batch, seq_len, hidden_size]`` tensor.

        Returns:
            ``[batch, num_tokens + seq_len, hidden_size]`` tensor.
        """
        batch_size = hidden_states.shape[0]
        soft = self.soft_tokens.expand(batch_size, -1, -1)
        soft = soft.to(hidden_states.device, hidden_states.dtype)
        return torch.cat([soft, hidden_states], dim=1)

    def get_params(self, lr: float) -> dict:
        """Return an optimizer parameter-group dict."""
        return {"params": list(self.parameters()), "lr": lr}


def create_vlm_ptuning(vlm, num_tokens: int, init_std: float = 0.02) -> SoftPromptEmbeddings:
    """Create a :class:`SoftPromptEmbeddings` sized for *vlm*'s input embeddings.

    Args:
        vlm: Vision-language model; used to infer the embedding dimension.
        num_tokens: Number of soft prompt tokens to prepend.
        init_std: Standard deviation for random initialisation.

    Returns:
        :class:`SoftPromptEmbeddings` instance (not moved to any device yet).
    """
    hidden_size = vlm.get_input_embeddings().embedding_dim
    module = SoftPromptEmbeddings(num_tokens, hidden_size, init_std)
    logger.info(
        "Created VLM P-tuning module: %d tokens, hidden_size=%d",
        num_tokens,
        hidden_size,
    )
    return module


def create_flux_encoder_ptuning(
    pipe, num_tokens: int, init_std: float = 0.02
) -> SoftPromptEmbeddings:
    """Create a :class:`SoftPromptEmbeddings` sized for Flux's T5 text encoder.

    The Flux transformer attends over ``encoder_hidden_states`` via
    cross-attention; prepending soft tokens there gives the denoiser extra
    learnable context.

    Args:
        pipe: Flux diffusion pipeline; ``pipe.text_encoder_2`` must be the T5
            encoder.
        num_tokens: Number of soft prompt tokens to prepend.
        init_std: Standard deviation for random initialisation.

    Returns:
        :class:`SoftPromptEmbeddings` instance (not moved to any device yet).
    """
    hidden_size = pipe.text_encoder_2.config.hidden_size
    module = SoftPromptEmbeddings(num_tokens, hidden_size, init_std)
    logger.info(
        "Created Flux encoder P-tuning module: %d tokens, hidden_size=%d",
        num_tokens,
        hidden_size,
    )
    return module


def apply_encoder_ptuning(prompt_kwargs: dict, ptuning: SoftPromptEmbeddings) -> dict:
    """Return a copy of *prompt_kwargs* with soft tokens prepended to the text
    encoder hidden states.

    Updates both ``encoder_hidden_states`` (shape
    ``[1, seq_len, hidden_size]`` → ``[1, num_tokens+seq_len, hidden_size]``)
    and ``txt_ids`` (shape ``[seq_len, 3]`` → ``[num_tokens+seq_len, 3]``)
    when present.

    Args:
        prompt_kwargs: Dict produced by :func:`dig_helpers.encode_flux_text`.
        ptuning: Trained (or initialised) soft-prompt module.

    Returns:
        Modified copy of *prompt_kwargs*.
    """
    prompt_kwargs = dict(prompt_kwargs)
    encoder_hs = prompt_kwargs.get("encoder_hidden_states")
    if encoder_hs is not None:
        prompt_kwargs["encoder_hidden_states"] = ptuning.prepend(encoder_hs)
        # Extend txt_ids with zero-filled rows for the soft tokens
        txt_ids = prompt_kwargs.get("txt_ids")
        if txt_ids is not None:
            n = ptuning.num_tokens
            prefix_ids = torch.zeros(
                n, txt_ids.shape[-1],
                device=txt_ids.device,
                dtype=txt_ids.dtype,
            )
            prompt_kwargs["txt_ids"] = torch.cat([prefix_ids, txt_ids], dim=0)
    return prompt_kwargs
