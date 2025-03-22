import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData

# Dummy graph data (replace with your actual data)
data = HeteroData()
data['customer'].x = torch.randn(1, 3)  # [1 customer, 3 features: age, income, gender]
data['order'].x = torch.randn(5, 2)     # [5 orders, 2 features: status embedding, time]
data['product'].x = torch.randn(10, 4)  # [10 products, 4 features]
data['customer', 'places', 'order'].edge_index = torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]).long()
data['order', 'contains', 'product'].edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]).long()

class GraphAttentionVAE(nn.Module):
    def __init__(self, customer_dim, order_dim, product_dim, hidden_dim, latent_dim):
        super(GraphAttentionVAE, self).__init__()
        
        # Define heterogeneous GNN layers with attention (GAT)
        self.conv1 = HeteroConv({
            ('customer', 'places', 'order'): GATConv((customer_dim, order_dim), hidden_dim, heads=4, dropout=0.2),
            ('order', 'contains', 'product'): GATConv((order_dim, product_dim), hidden_dim, heads=4, dropout=0.2),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('customer', 'places', 'order'): GATConv((hidden_dim * 4, hidden_dim * 4), hidden_dim, heads=1, dropout=0.2),
            ('order', 'contains', 'product'): GATConv((hidden_dim * 4, hidden_dim * 4), hidden_dim, heads=1, dropout=0.2),
        }, aggr='mean')

        # VAE encoder: map to latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution

        # VAE decoder: reconstruct edge probabilities
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output edge probability
            nn.Sigmoid()
        )

    def encode(self, x_dict, edge_index_dict):
        # First GNN layer with attention
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second GNN layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Customer embedding after aggregating order info
        customer_embed = x_dict['customer']
        
        # Estimate latent distribution parameters
        mu = self.fc_mu(customer_embed)
        logvar = self.fc_logvar(customer_embed)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, product_embed):
        # Concatenate customer latent vector (z) with product embeddings
        z_expanded = z.repeat(product_embed.size(0), 1)  # Match product batch size
        combined = torch.cat([z_expanded, product_embed], dim=-1)
        edge_prob = self.decoder(combined)
        return edge_prob

    def forward(self, x_dict, edge_index_dict, product_embed):
        # Encode customer to latent space
        mu, logvar = self.encode(x_dict, edge_index_dict)
        
        # Reparameterize to sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode to predict edge probabilities with products
        edge_prob = self.decode(z, product_embed)
        return edge_prob, mu, logvar

# Loss function: Reconstruction loss + KL divergence
def vae_loss(edge_pred, edge_true, mu, logvar):
    recon_loss = F.binary_cross_entropy(edge_pred, edge_true, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training setup
customer_dim, order_dim, product_dim = 3, 2, 4
hidden_dim, latent_dim = 64, 32
model = GraphAttentionVAE(customer_dim, order_dim, product_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy edge labels (replace with real customer-product edge data)
edge_true = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32)  # 10 products, binary labels

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    edge_pred, mu, logvar = model(data.x_dict, data.edge_index_dict, data['product'].x)
    
    # Compute loss
    loss = vae_loss(edge_pred.squeeze(), edge_true, mu, logvar)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Recommendation
model.eval()
with torch.no_grad():
    edge_pred, _, _ = model(data.x_dict, data.edge_index_dict, data['product'].x)
    top_k = torch.topk(edge_pred.squeeze(), k=3).indices  # Top-3 product recommendations
    print("Recommended product indices:", top_k.tolist())
