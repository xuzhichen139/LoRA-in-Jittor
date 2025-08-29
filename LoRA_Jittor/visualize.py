import numpy as np
import matplotlib.pyplot as plt
import re
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
#提取文件
def get_log_data(log_path):
    epochs = []
    losses = []
    accs = []

    pattern = re.compile(r"Epoch (\d+)/\d+ \| Loss: (\d+\.\d+) \| Acc: (\d+\.\d+)")
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))
    return epochs, losses, accs


#Loss曲线对比图
def plot_loss_curve():
    jt_epochs, jt_losses, _ = get_log_data("logs/jittor_train.log")
    pt_epochs, pt_losses, _ = get_log_data("/mnt/c/Users/13913/Desktop/LoRA_Pytorch/logs/pytorch_train.log")

    plt.figure(figsize=(8, 5))
    #Jittor Loss曲线
    plt.plot(jt_epochs, jt_losses, label="Jittor Loss", color="blue", linewidth=2, marker="o")
    #PyTorch Loss曲线
    plt.plot(pt_epochs, pt_losses, label="PyTorch Loss", color="orange", linewidth=2, marker="s", linestyle="--")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Jittor vs PyTorch Loss ", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    #保存图片
    plt.savefig("figures/loss_curve.png", dpi=300)
    plt.close()
    print("Loss曲线已保存到figures/loss_curve.png")


#准确率对比图
def plot_acc_curve():

    jt_epochs, _, jt_accs = get_log_data("logs/jittor_train.log")
    pt_epochs, _, pt_accs = get_log_data("/mnt/c/Users/13913/Desktop/LoRA_Pytorch/logs/pytorch_train.log")

    plt.figure(figsize=(8, 5))
    plt.plot(jt_epochs, jt_accs, label="Jittor Acc", color="blue", linewidth=2, marker="o")
    plt.plot(pt_epochs, pt_accs, label="PyTorch Acc", color="orange", linewidth=2, marker="s", linestyle="--")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Jittor vs PyTorch", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.savefig("figures/acc_curve.png", dpi=300)
    plt.close()
    print("准确率对比图已保存到figures/acc_curve.png")


#混淆矩阵对比
def plot_confusion_matrix():
    #Jittor预测结果
    jt_data = np.load("results/jittor_predictions.npz")
    jt_preds = jt_data["preds"]
    jt_labels = jt_data["labels"]
    #PyTorch预测结果
    pt_data = np.load("/mnt/c/Users/13913/Desktop/LoRA_Pytorch/results/pytorch_predictions.npz")
    pt_preds = pt_data["preds"]
    pt_labels = pt_data["labels"]

    #计算混淆矩阵
    def get_cm(preds, labels):
        #混淆矩阵：[[正确预测0的数量, 错误预测1的数量], [错误预测0的数量, 正确预测1的数量]]
        cm = [[0, 0], [0, 0]]
        for p, l in zip(preds, labels):
            if l == 0:
                if p == 0:
                    cm[0][0] += 1
                else:
                    cm[0][1] += 1
            else:
                if p == 0:
                    cm[1][0] += 1
                else:
                    cm[1][1] += 1
        return np.array(cm)

    jt_cm = get_cm(jt_preds, jt_labels)
    pt_cm = get_cm(pt_preds, pt_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #Jittor混淆矩阵
    im1 = ax1.imshow(jt_cm, cmap="Blues")
    ax1.set_title("Jittor Confusion Matrix", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, jt_cm[i, j], ha="center", va="center")
    #PyTorch 混淆矩阵
    im2 = ax2.imshow(pt_cm, cmap="Blues")
    ax2.set_title("PyTorch Confusion Matrix", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, pt_cm[i, j], ha="center", va="center")

    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()
    print("混淆矩阵对比图已保存到figures/confusion_matrix.png")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_loss_curve()
    plot_acc_curve()
    plot_confusion_matrix()
    print("所有图片已生成！")