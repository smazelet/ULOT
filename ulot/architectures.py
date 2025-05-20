import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear, Sequential
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch.nn import ReLU


class matching_layer(nn.Module):
    """
    Matching layer
    """
    def __init__(
        self, temperature_value=3.0, in_channels=None, d_alpha_enc=6
    ):
        """"
        intialize_scale: float
           Temperature for the softmax
        in_channels: int
           Input feature dimension 
        d_pos_enc: int
            Dimension of the alpha encoding
        """
        super(matching_layer, self).__init__()
        self.a = nn.Parameter(torch.tensor([10.0]), requires_grad=False)
        self.linear_volume = Linear(in_channels + 1 + d_alpha_enc, 1)
        self.linear = nn.Sequential(
            Linear(in_channels + 1 + d_alpha_enc, in_channels + 1 + d_alpha_enc),
            nn.ReLU(),
            Linear(in_channels + 1 + d_alpha_enc, in_channels + 1 + d_alpha_enc),
        )

    def forward(self, x1, x2, batch1, batch2, x1_rho_alpha, x2_rho_alpha):
        x1, mask1 = to_dense_batch(x1, batch1)
        x2, mask2 = to_dense_batch(x2, batch2)
        x1_rho_alpha, _ = to_dense_batch(x1_rho_alpha, batch1)
        x2_rho_alpha, _ = to_dense_batch(x2_rho_alpha, batch2)

        x1_rho_alpha_lin = self.linear(x1_rho_alpha)
        x2_rho_alpha_lin = self.linear(x2_rho_alpha)

        x1_norm = F.normalize(x1_rho_alpha_lin, p=2, dim=-1)
        x2_norm = F.normalize(x2_rho_alpha_lin, p=2, dim=-1)

        C = torch.bmm(x1_norm, x2_norm.transpose(1, 2))

        # replace masked elements in the similarity matrix by -inf before appplying softmax
        C[~mask1.unsqueeze(2).expand_as(C)] = float("-inf")
        C[~mask2.unsqueeze(1).expand_as(C)] = float("-inf")

        a = self.a**2

        attention1 = torch.softmax(a * C, dim=2)
        attention2 = torch.softmax(a * C, dim=1)

        # masked lines become Nans, replace them by zeros
        attention1 = torch.nan_to_num(attention1, nan=0.0)
        attention2 = torch.nan_to_num(attention2, nan=0.0)
        x1_match = x1 - torch.bmm(attention1, x2)
        x2_match = x2 - torch.bmm(attention2.transpose(1, 2), x1)

        x1_match = x1_match * mask1.unsqueeze(-1)
        x2_match = x2_match * mask2.unsqueeze(-1)

        num_nodes1 = mask1.sum(dim=1, keepdim=True).unsqueeze(-1)
        num_nodes2 = mask2.sum(dim=1, keepdim=True).unsqueeze(-1)

        #volume on the nodes
        x1_volume = torch.sigmoid(self.linear_volume(x1_rho_alpha))
        x2_volume = torch.sigmoid(self.linear_volume(x2_rho_alpha))
        attention1 = attention1 * x1_volume
        attention2 = attention2 * x2_volume.permute(0, 2, 1)
        matching = (attention1 / num_nodes1 + attention2 / num_nodes2) / 2
        x1_match = x1_match[mask1]
        x2_match = x2_match[mask2]
        return x1_match, x2_match, matching, mask1, mask2



class GMN_layer(nn.Module):
    """
    Undividual ULOT layer
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        hidden_channels_gcn,
        out_channels,
        temperature_value,
        d_alpha_enc=6,
    ):
        """
        in_channels: int
            Input feature dimension
        hidden_channels: int
            Hidden feature dimension
        hidden_channels_gcn: int
            Hidden feature dimension for the GCN
        out_channels: int
            Output feature dimension
        temperature_value: float
            Temperature for the softmax
        learn_volume: bool
            Whether to learn the volume
        d_alpha_enc: int
            Dimension of the alpha encoding
        """
        super().__init__()
        self.linear = Linear(in_channels, hidden_channels_gcn)
        self.linear_match = Linear(in_channels, hidden_channels_gcn)
        self.match = matching_layer(
            temperature_value, in_channels, d_alpha_enc
        )
        self.message = Sequential(
            " x,  edge_index,",
            [
                (
                    GCNConv(in_channels + 1 + d_alpha_enc, hidden_channels_gcn),
                    "x, edge_index -> x",
                ),
                ReLU(inplace=True),
                (
                    GCNConv(hidden_channels_gcn, hidden_channels_gcn),
                    "x, edge_index -> x",
                ),
            ],
        )
        self.tail = Sequential(
            " x",
            [
                (
                    Linear(3 * hidden_channels_gcn + 1 + d_alpha_enc, hidden_channels),
                    "x-> x",
                ),
                ReLU(inplace=True),
                (Linear(hidden_channels, out_channels), "x-> x"),
            ],
        )
        self.in_channels = in_channels
        self.d_pos_enc = d_alpha_enc

    def forward(
        self, x1, x2, edge1, edge2, x1_batch, x2_batch, rhos_s, rhos_t, alpha_s, alpha_t
    ):
        x1_rho_alpha = torch.cat([alpha_s, rhos_s, x1], dim=1)
        x2_rho_alpha = torch.cat([alpha_t, rhos_t, x2], dim=1)

        x1_message = self.message(x1_rho_alpha, edge1)
        x2_message = self.message(x2_rho_alpha, edge2)

        x1_match, x2_match, P, mask1, mask2 = self.match(
            x1, x2, x1_batch, x2_batch, x1_rho_alpha, x2_rho_alpha
        )

        x1 = self.linear(x1)
        x2 = self.linear(x2)

        x1_match = self.linear_match(x1_match)
        x2_match = self.linear_match(x2_match)

        x1 = self.tail(torch.cat([x1, x1_message, x1_match, rhos_s, alpha_s], dim=1))
        x2 = self.tail(torch.cat([x2, x2_message, x2_match, rhos_t, alpha_t], dim=1))
        return x1, x2, P, mask1, mask2


class ULOT_net(nn.Module):
    """
    ULOT network
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        hidden_channels_message,
        out_channels,
        num_layers,
        intialize_scale=3.0,
        d_alpha_enc=6,
    ):
        """
        in_channels: int
            Input feature dimension
        hidden_channels: int
            Hidden feature dimension
        hidden_channels_message: int
            Hidden feature dimension for the message passing
        out_channels: int
            Output feature dimension
        num_layers: int
            Number of layers        
        intialize_scale: float
            Temperature for the softmax 
        d_alpha_enc: int
            Dimension of the alpha encoding
        """
        super().__init__()

        self.first_GMN_layer = GMN_layer(
            in_channels,
            hidden_channels,
            hidden_channels_message,
            out_channels,
            intialize_scale,
            d_alpha_enc,
        )
        self.GMN_layer = GMN_layer(
            out_channels,
            hidden_channels,
            hidden_channels_message,
            out_channels,
            intialize_scale,
            d_alpha_enc,
        )

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.d_pos_enc = d_alpha_enc
        self.x_pos_enc = d_alpha_enc

    def forward(
        self,
        x1,
        x2,
        edge1,
        edge2,
        x1_batch,
        x2_batch,
        x1_rho,
        x2_rho,
        x1_alpha,
        x2_alpha,
    ):
        x1_alpha = alpha_encoding(x1_alpha, self.d_pos_enc)
        x2_alpha = alpha_encoding(x2_alpha, self.d_pos_enc)
        x1, x2, P, mask1, mask2 = self.first_GMN_layer(
            x1, x2, edge1, edge2, x1_batch, x2_batch, x1_rho, x2_rho, x1_alpha, x2_alpha
        )
        for _ in range(self.num_layers):
            x1, x2, P, mask1, mask2 = self.GMN_layer(
                x1,
                x2,
                edge1,
                edge2,
                x1_batch,
                x2_batch,
                x1_rho,
                x2_rho,
                x1_alpha,
                x2_alpha,
            )
        return P, mask1, mask2


def alpha_encoding(alpha,d):
    """
    Positional encoding for the alpha values
    alpha: torch.Tensor
        Alpha values
    d: int
        Dimension of the positional encoding
    """
    d_half = d // 2
    N = len(alpha)
    j_values = torch.arange(d, device=alpha.device)
    alphas = alpha[:, None]

    cos_part = torch.cos(torch.pi * j_values[:d_half, None] *  alphas) 
    sin_part = torch.sin(torch.pi * j_values[d_half:, None] * (1 -  alphas)) 
    values = torch.empty((N, d), device=alpha.device)  
    values[:, :d_half] = cos_part.squeeze() 
    values[:, d_half:] = sin_part.squeeze() 
    return values