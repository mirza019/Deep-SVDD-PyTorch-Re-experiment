import os, sys
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

# Add project root (two levels up) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torchvision.utils import save_image

# =================== CONFIG ===================
LOG_BASE = "log/mnist_reexp"
MODEL_PATH = os.path.join(LOG_BASE, "model.tar")
LOG_PATH = os.path.join(LOG_BASE, "log.txt")
DEVICE = "cpu"   # change to "cuda" if using GPU
SAVE_DIR = "imgs/visuals"
os.makedirs(SAVE_DIR, exist_ok=True)
# ==============================================


# 1️⃣ Extract & Smooth Training Loss
def extract_epoch_losses(logfile):
    losses_per_epoch = {}
    with open(logfile) as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/\d+\s+Time: [\d\.]+\s+Loss: ([\d\.E-]+)", line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                losses_per_epoch.setdefault(epoch, []).append(loss)
    epochs = sorted(losses_per_epoch.keys())
    avg_losses = [np.mean(losses_per_epoch[e]) for e in epochs]
    return epochs, avg_losses


def plot_loss_curve():
    e, l = extract_epoch_losses(LOG_PATH)
    window = 5
    smooth = np.convolve(l, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(6, 4))
    plt.plot(e[:len(smooth)], smooth, label="Hybrid Deep SVDD + AE")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Smoothed Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.close()
    print("✅ Saved: loss_curve.png")


# 2️⃣ t-SNE Latent Space Visualization
def plot_latent_space(deep_SVDD, test_loader):
    deep_SVDD.net.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            x, y, _ = data
            x = x.to(DEVICE)
            z = deep_SVDD.net(x)
            embeddings.append(z.cpu().numpy())
            labels.append(y.numpy())

    X = np.concatenate(embeddings)
    Y = np.concatenate(labels)
    tsne = TSNE(n_components=2, random_state=42)
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=Y, cmap='coolwarm', s=4)
    plt.title("t-SNE: Latent Space (Normal vs Anomaly)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "tsne_latent.png"))
    plt.close()
    print("✅ Saved: tsne_latent.png")


# 3️⃣ Reconstruction Visualization
def plot_reconstruction(ae_net, test_loader):
    ae_net.eval()
    dataiter = iter(test_loader)
    images, _, _ = next(dataiter)
    with torch.no_grad():
        z, recon = ae_net(images)

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(10):
        axes[0][i].imshow(images[i, 0].cpu(), cmap="gray")
        axes[1][i].imshow(recon[i, 0].detach().cpu(), cmap="gray")
        axes[0][i].axis("off")
        axes[1][i].axis("off")
    axes[0][0].set_ylabel("Original")
    axes[1][0].set_ylabel("Reconstructed")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "reconstruction.png"))
    plt.close()
    print("✅ Saved: reconstruction.png")


# 4️⃣ ROC Curve & AUC Calculation
def plot_roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
    plt.close()
    print(f"✅ Saved: roc_curve.png  |  AUC = {roc_auc:.3f}")


# 5️⃣ Grad-CAM Example (LeNet conv2)
def plot_gradcam(deep_SVDD, test_loader):
    try:
        from src.utils.gradcam import GradCAM
        # Use final conv layer instead of .encoder
        gradcam = GradCAM(deep_SVDD.net, deep_SVDD.net.conv2)
        for img, _ in test_loader:
            cam = gradcam(img)
            plt.imshow(img[0, 0].cpu(), cmap='gray')
            plt.imshow(cam[0], cmap='jet', alpha=0.5)
            plt.axis("off")
            plt.title("Grad-CAM: Anomaly Focus Map")
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, "gradcam.png"))
            plt.close()
            print("✅ Saved: gradcam.png")
            break
    except Exception as e:
        print(f"⚠️  Grad-CAM skipped: {e}")


# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    from src.deepSVDD import DeepSVDD
    from src.datasets.main import load_dataset

    print("🔹 Loading dataset and model...")
    dataset = load_dataset("mnist", "data", normal_class=0)
    _, test_loader = dataset.loaders(batch_size=128, num_workers=0)

    from src.networks.main import build_network

    deep_SVDD = DeepSVDD(objective='one-class', nu=0.1)
    deep_SVDD.set_network("mnist_LeNet")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    deep_SVDD.net.load_state_dict(checkpoint['net_dict'])

    print("🔹 Generating visualizations...")
    plot_loss_curve()
    plot_latent_space(deep_SVDD, test_loader)

    if hasattr(deep_SVDD, "ae_net") and deep_SVDD.ae_net is not None:
        plot_reconstruction(deep_SVDD.ae_net, test_loader)
    else:
        print("⚠️  No autoencoder found in model — skipping reconstruction plot.")

    try:
        plot_gradcam(deep_SVDD, test_loader)
    except Exception as e:
        print(f"⚠️  Grad-CAM skipped: {e}")

    print("✅ All visualizations completed! Files saved in:", SAVE_DIR)
