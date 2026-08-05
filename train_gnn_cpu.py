import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from gnn_model import WriterHybridGAT

CACHE_FILE = "cvl_graph_cache.pt"

class CachedGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["x"], item["adj"], item["mask"], item["classical"], item["label"], item["page_num"]

def apply_graph_augmentation(x, mask):
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(x[:, :, :2]) * 0.015
        x[:, :, :2] = (x[:, :, :2] + noise * mask.unsqueeze(-1)).clamp(0.0, 1.0)
    return x

def evaluate(model, dataloader, device):
    model.eval()
    page_predictions = {}

    with torch.no_grad():
        for x, adj, mask, classical, labels, pages in dataloader:
            x, adj, mask, classical, labels = (
                x.to(device), adj.to(device), mask.to(device), 
                classical.to(device), labels.to(device)
            )
            logits, _ = model(x, adj, mask, classical)
            probs = torch.softmax(logits, dim=1)

            for i in range(x.size(0)):
                key = (labels[i].item(), pages[i].item())
                if key not in page_predictions:
                    page_predictions[key] = torch.zeros(310, device=device)
                page_predictions[key] += probs[i]

    page_correct = 0
    for (true_label, _), avg_probs in page_predictions.items():
        if torch.argmax(avg_probs).item() == true_label:
            page_correct += 1

    return (page_correct / len(page_predictions)) * 100 if len(page_predictions) > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Accelerator Device: {device}")

    cached_data = torch.load(CACHE_FILE)
    
    # Train on full Pages 1, 2, 3 | Test on Pages 4 & 6
    full_train_list = [item for item in cached_data if item["page_num"] in [1, 2, 3]]
    test_list = [item for item in cached_data if item["page_num"] in [4, 6]]

    # Create 90/10 Train/Val Split from Pages 1-3
    val_size = int(len(full_train_list) * 0.10)
    train_size = len(full_train_list) - val_size
    
    train_dataset, val_dataset = random_split(
        CachedGraphDataset(full_train_list), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = CachedGraphDataset(test_list)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Train samples (Pages 1-3 90%): {train_size}")
    print(f"Val samples (Pages 1-3 10%): {val_size}")
    print(f"Test samples (Pages 4 & 6): {len(test_list)}")

    model = WriterHybridGAT(node_dim=7, classical_dim=11, hidden_dim=128, num_classes=310).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print("\nStarting GAT Training Loop (Pages 1-3)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, adj, mask, classical, labels, _ in train_loader:
            x, adj, mask, classical, labels = (
                x.to(device), adj.to(device), mask.to(device), 
                classical.to(device), labels.to(device)
            )
            
            x = apply_graph_augmentation(x, mask)
            
            optimizer.zero_grad()
            logits, _ = model(x, adj, mask, classical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_gat.pt")
            saved_str = " [BEST VAL CHECKPOINT SAVED]"
        else:
            saved_str = ""

        print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/train_size:.4f} | Val Page Acc: {val_acc:.2f}%{saved_str}")

    print("\n==================================================")
    print("FINISHED TRAINING. EVALUATING TEST SET (Pages 4 & 6)...")
    print("==================================================")
    model.load_state_dict(torch.load("best_hybrid_gat.pt"))
    final_test_acc = evaluate(model, test_loader, device)
    
    print(f"\n FINAL UNBIASED TEST PAGE ACCURACY: {final_test_acc:.2f}%\n")

if __name__ == "__main__":
    main()
