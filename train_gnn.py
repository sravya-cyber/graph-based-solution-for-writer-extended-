# Paste the full contents of your local train_gnn.py here
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gnn_model import WriterGNN

CACHE_FILE = "cvl_graph_cache.pt"

class CachedGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["x"], item["adj"], item["mask"], item["label"], item["page_num"]

def main():
    print(f"Loading cached graph dataset from '{CACHE_FILE}'...")
    cached_data = torch.load(CACHE_FILE)
    
    train_data = [item for item in cached_data if item["page_num"] in [1, 2, 3]]
    test_data = [item for item in cached_data if item["page_num"] in [4, 6]]

    train_dataset = CachedGraphDataset(train_data)
    test_dataset = CachedGraphDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = WriterGNN(in_dim=2, hidden_dim=64, num_classes=310)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 20
    print("\nStarting CPU Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, adj, mask, labels, _ in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x, adj, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)

        train_acc = (correct / total) * 100

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, adj, mask, labels, _ in test_loader:
                logits, _ = model(x, adj, mask)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += x.size(0)

        test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0.0
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
