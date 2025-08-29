import torch
import logging
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shutil

def init_logger():
    log_path = "logs/pytorch_train.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PyTorch - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


#LoRA
class LoRAModule(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.scale = rank ** 0.5
        self.W_A = nn.Linear(in_dim, rank, bias=False)
        self.W_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.W_A.weight, std=0.01)
        nn.init.zeros_(self.W_B.weight)

    def forward(self, x):
        return self.W_B(self.W_A(x)) * self.scale


#带LoRA的多头注意力层
class LoRA_MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, rank=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            proj.weight.requires_grad = False
        self.q_lora = LoRAModule(embed_dim, embed_dim, rank=rank)
        self.v_lora = LoRAModule(embed_dim, embed_dim, rank=rank)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.weight.requires_grad = False

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x) + self.q_lora(x)
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_lora(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)


#Transformer编码器
class SimplifiedTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, ff_dim=128, rank=4):
        super().__init__()
        self.attention = LoRA_MultiHeadAttention(embed_dim, num_heads, rank)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                layer.weight.requires_grad = False
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


#训练
def main():

    torch.manual_seed(42)
    np.random.seed(42)

    #清空文件夹
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)

    logger = init_logger()

    #超参数
    embed_dim = 64
    seq_len = 10
    vocab_size = 100
    num_classes = 2
    batch_size = 8
    epochs = 1000
    rank = 4

    logger.info(f"训练开始！超参数：batch_size={batch_size}, lr=1e-3, epochs={epochs}, rank={rank}")

    x = torch.randint(0, vocab_size, (batch_size *20 , seq_len))
    y = torch.randint(0, num_classes, (batch_size *20,))
    embedding = nn.Embedding(vocab_size, embed_dim)
    embedding.weight.requires_grad = False

    #OutputProcessor
    class OutputProcessor(nn.Module):
        def forward(self, x):
            x = x.transpose(1, 2)
            x = nn.AdaptiveAvgPool1d(1)(x)
            x = x.squeeze(-1)
            return x

    #完整模型
    model = nn.Sequential(
        embedding,
        SimplifiedTransformer(embed_dim, rank=rank),
        OutputProcessor(),
        nn.Linear(embed_dim, num_classes)
    )
    for param in model[-1].parameters():
        param.requires_grad = True

    #优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    #损失函数
    criterion = nn.CrossEntropyLoss()

    #训练
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(len(x))[:batch_size]
        batch_x, batch_y = x[idx], y[idx]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            acc = (outputs.argmax(dim=1) == batch_y).float().mean().item()
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    logger.info("训练结束！PyTorch 模型训练完成。")
    # 保存预测结果
    test_x = x[-batch_size:]
    test_y = y[-batch_size:]
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_x)
    pytorch_preds = test_outputs.argmax(dim=1).numpy().squeeze().flatten()[:batch_size]
    pytorch_labels = test_y.numpy().squeeze().flatten()[:batch_size]
    # 保存
    np.savez(
        "results/pytorch_predictions.npz",
        preds=pytorch_preds,
        labels=pytorch_labels
    )
    logger.info("预测结果已保存到results/pytorch_predictions.npz")


if __name__ == "__main__":
    main()