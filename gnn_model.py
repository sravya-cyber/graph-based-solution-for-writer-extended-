import torch
import torch.nn as nn
import torch.nn.functional as F

class GATv2Layer(nn.Module):
    """Graph Attention Network v2 Layer"""
    def __init__(self, in_dim, out_dim, heads=2):
        super(GATv2Layer, self).__init__()
        self.heads = heads
        self.out_dim = out_dim // heads
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, adj):
        B, N, C = x.size()
        h = self.fc(x).view(B, N, self.heads, self.out_dim)
        
        # Calculate pair-wise attention scores
        attn_logits = (h * self.attn).sum(dim=-1)
        attn_logits = attn_logits.unsqueeze(2) + attn_logits.unsqueeze(1)
        
        # Mask non-adjacent nodes
        adj_expanded = adj.unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(adj_expanded == 0, -1e9)
        attn_weights = F.softmax(attn_logits, dim=2)
        
        # Message aggregation
        out = torch.einsum('bnmh,bmhd->bnhd', attn_weights, h)
        out = out.reshape(B, N, -1)
        
        out = self.bn(out.view(B * N, -1)).view(B, N, -1)
        return F.elu(out)

class WriterHybridGAT(nn.Module):
    def __init__(self, node_dim=7, classical_dim=11, hidden_dim=128, num_classes=310):
        super(WriterHybridGAT, self).__init__()
        self.gat1 = GATv2Layer(node_dim, hidden_dim, heads=2)
        self.gat2 = GATv2Layer(hidden_dim, hidden_dim, heads=2)
        self.gat3 = GATv2Layer(hidden_dim, hidden_dim, heads=2)

        self.attn_pool = nn.Linear(hidden_dim, 1)

        combined_dim = hidden_dim + classical_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, adj, mask, classical_feats):
        h = self.gat1(x, adj)
        h = self.gat2(h, adj)
        h = self.gat3(h, adj)

        # Masked Attention pooling
        attn_w = self.attn_pool(h)
        attn_w = attn_w.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_w = F.softmax(attn_w, dim=1)

        graph_emb = torch.sum(h * attn_w, dim=1)
        combined = torch.cat([graph_emb, classical_feats], dim=1)

        logits = self.classifier(combined)
        return logits, combined
