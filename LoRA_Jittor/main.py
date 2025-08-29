import jittor as jt
import jittor.nn as nn
import numpy as np
import logging
import os
import shutil

#保存文件
def init_logger():
    log_path = "logs/jittor_train.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - Jittor - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

#LoRA
class LoRAModule(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.scale = rank ** 0.5
        self.W_A = jt.nn.Parameter(jt.normal(0.0, 0.01, (rank, in_dim)))
        self.W_B = jt.nn.Parameter(jt.zeros((out_dim, rank)))
    def execute(self, x):
        lora_a = jt.matmul(x, self.W_A.transpose(0, 1))
        lora_out = jt.matmul(lora_a, self.W_B.transpose(0, 1))
        return lora_out * self.scale

#多头注意力层
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

    def execute(self, x):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x) + self.q_lora(x)
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_lora(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = nn.softmax(attn_scores, dim=-1)
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

    def execute(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OutputProcessor(nn.Module):
    def execute(self, x):
        x = jt.mean(x, dim=1)
        return x

#训练
def main():
    jt.seed(42)
    np.random.seed(42)
    # 清空文件夹
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)

    logger = init_logger()
    #超参数
    jt.flags.use_cuda = 0
    embed_dim = 64
    seq_len = 10
    vocab_size = 100
    num_classes = 2
    batch_size = 8
    epochs = 1000
    rank = 4

    #生成数据
    np_x = np.random.randint(0, vocab_size, (batch_size *20, seq_len))
    np_y = np.random.randint(0, num_classes, (batch_size *20,))
    x = jt.array(np_x)
    y = jt.array(np_y)

    #模型定义
    embedding = nn.Embedding(vocab_size, embed_dim)
    embedding.weight.requires_grad = False
    model = nn.Sequential(embedding, SimplifiedTransformer(embed_dim, rank=rank), OutputProcessor(), nn.Linear(embed_dim, num_classes))
    for param in model[-1].parameters():
        param.requires_grad = True

    #优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = nn.Adam(trainable_params, lr=1e-3)
    #损失函数
    criterion = nn.CrossEntropyLoss()

    logger.info(f"训练开始！超参数：batch_size={batch_size}, lr=1e-3, epochs={epochs}")

    #训练循环
    model.train()
    for epoch in range(epochs):
        start = epoch * batch_size
        end = start + batch_size
        idx = jt.randperm(len(x))[:batch_size]
        batch_x = x[idx]
        batch_y = y[idx]

        #前向计算
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        #反向更新
        optimizer.zero_grad()
        loss = criterion(outputs, batch_y)
        optimizer.step(loss)

        #计算准确率
        preds_np = np.asarray(outputs.argmax(dim=1)).squeeze().flatten()[:batch_size]
        labels_np = np.asarray(batch_y).squeeze().flatten()[:batch_size]
        acc = (preds_np == labels_np).sum() / batch_size

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    logger.info("训练结束！")

    test_x = x[-batch_size:]
    test_y = y[-batch_size:]
    model.eval()
    with jt.no_grad():
        test_outputs = model(test_x)

    jittor_preds = np.asarray(test_outputs.argmax(dim=1)).squeeze().flatten()[:batch_size]
    jittor_labels = np.asarray(test_y).squeeze().flatten()[:batch_size]
    # 保存
    np.savez(
        "results/jittor_predictions.npz",
        preds=jittor_preds,
        labels=jittor_labels
    )
    logger.info("预测结果已保存到results/jittor_predictions.npz")

if __name__ == "__main__":
    main()