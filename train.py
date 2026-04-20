"""Train TeaLeafNet. Usage: python train.py --data_dir data/tea_leaf"""
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from src.model import TeaLeafNet
from src.dataset import build_loaders, CLASSES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/tea_leaf")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tr_loader, va_loader = build_loaders(args.data_dir, args.batch_size)
    model     = TeaLeafNet(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); tr_loss = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in va_loader:
                preds = model(imgs.to(device)).argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)
        acc = correct / total
        print(f"Epoch {epoch:3d} | loss={tr_loss/len(tr_loader):.4f} | acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            print(f"  ✅ Saved best model (acc={best_acc:.4f})")

if __name__ == "__main__": main()
