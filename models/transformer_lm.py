import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerLMConfig:
    vocab_size: int = 1024
    block_size: int = 224
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerLMConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        causal_mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    def __init__(self, config: TransformerLMConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerLMConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerLMConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Input sequence length {T} exceeds block_size {self.config.block_size}"
            )

        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()

        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.config.block_size:]

            outputs = self(input_cond)
            logits = outputs["logits"]

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=224)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    config = TransformerLMConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = TransformerLM(config)

    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.block_size),
        dtype=torch.long,
    )

    labels = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.block_size),
        dtype=torch.long,
    )

    outputs = model(input_ids, labels=labels)

    print(f"Vocab size      : {config.vocab_size}")
    print(f"Block size      : {config.block_size}")
    print(f"Layers          : {config.n_layer}")
    print(f"Heads           : {config.n_head}")
    print(f"Embedding dim   : {config.n_embd}")
    print(f"Num parameters  : {model.num_parameters():,}")
    print(f"Input shape     : {input_ids.shape}")
    print(f"Logits shape    : {outputs['logits'].shape}")
    print(f"Loss            : {outputs['loss'].item():.4f}")

    generated = model.generate(
        input_ids[:, :10],
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
    )

    print(f"Generated shape : {generated.shape}")


if __name__ == "__main__":
    main()
