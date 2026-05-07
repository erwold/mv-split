import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging
import logging

# Configure local logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Reduce noise from transformers library
transformers_logging.set_verbosity_error()

class Qwen3TextEncoder(nn.Module):
    """
    A text encoder wrapper for Qwen3 models (e.g., Qwen/Qwen3-0.6B).
    It uses AutoModel to be compatible with various transformer architectures.
    """
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device="cuda", dtype=torch.bfloat16, max_length=256):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length

        # 1. Load Tokenizer using AutoTokenizer for automatic compatibility
        log.info(f"Loading Tokenizer for: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token is set (a good practice)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = max_length

        # 2. Load Model using AutoModel to get the base transformer
        log.info(f"Loading Transformer Model: {model_name} on {device} with dtype {dtype}")
        self.text_model = AutoModel.from_pretrained(model_name).to(device=self.device, dtype=self.dtype)
        
        # Freeze parameters and set to evaluation mode
        self.text_model.eval()
        self.text_model.requires_grad_(False)
        
        self.hidden_size = self.text_model.config.hidden_size
        log.info(f"Qwen3TextEncoder initialized and frozen. Hidden Size: {self.hidden_size}")

    @torch.no_grad()
    def encode(self, texts: list[str]):
        """
        Encodes a batch of text prompts into last hidden state embeddings.
        """
        # Tokenization
        tokens = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        inputs_embeds = self.text_model.embed_tokens(input_ids)

        # Encoding with automatic mixed precision
        device_type = self.device.type.split(':')[0]
        with torch.autocast(device_type=device_type, dtype=self.dtype, enabled=True):
            outputs = self.text_model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        
        # (Batch, SequenceLength, HiddenDimension)
        last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state, attention_mask

    def forward(self, texts: list[str]):
        return self.encode(texts)