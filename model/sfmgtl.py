import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from model.tgcn import TGCN
import pickle
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import BatchNorm, GraphNorm
import networkx as nx


class DenseGATConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GATConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        # TODO Add support for edge features.
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, 1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x = self.lin(x).view(B, N, H, C)  # [B, N, H, C]

        alpha_src = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer=1):
        super(GNN, self).__init__()
        self.gnn1 = DenseGATConv(in_channels, hidden_channels, heads=2)
        self.bn1 = GraphNorm(hidden_channels)
        self.gnn2 = DenseGATConv(hidden_channels, out_channels, heads=2)
        self.bn2 = GraphNorm(out_channels)

    def forward(self, x, adj, mask=None):
        batch_size, node_num, _ = x.shape
        x = self.gnn1(x, adj, mask)
        x = self.bn1(x)
        x = self.gnn2(x, adj, mask)
        x = self.bn2(x)

        return x


def dense_diff_pool(x, adj, s, mask=None, normalize: bool = True):
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened adjacency matrix and
    two auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and (2) the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        normalize (bool, optional): If set to :obj:`False`, the link
            prediction loss is not divided by :obj:`adj.numel()`.
            (default: :obj:`True`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss


class DiffPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes1=100, num_nodes2=10):
        super(DiffPool, self).__init__()
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes1)
        self.gnn1_embed = GNN(input_dim, hidden_dim, input_dim)

        self.gnn2_pool = GNN(input_dim, hidden_dim, num_nodes2)
        self.gnn2_embed = GNN(input_dim, hidden_dim, input_dim)
        self.temperature = hidden_dim ** 0.5

    def forward(self, x, adj, origin_seq, mask=None):
        n2z_s = self.gnn1_pool(x, adj, mask)
        n2z_x = self.gnn1_embed(x, adj, mask)

        zone_x, adj_z, l1, e1 = dense_diff_pool(n2z_x, adj, n2z_s, mask)
        z2s_s = self.gnn2_pool(zone_x, adj_z, mask)
        z2s_x = self.gnn2_embed(zone_x, adj_z, mask)
        zone_temp = torch.bmm(torch.softmax(n2z_s, dim=-1).transpose(1, 2), origin_seq.transpose(1, 2)).transpose(1, 2)

        semantic_x, adj_s, l2, e2 = dense_diff_pool(z2s_x, adj_z, z2s_s)
        semantic_x = torch.mean(semantic_x, dim=1, keepdim=True)

        semantic_temp = torch.bmm(torch.softmax(z2s_s, dim=-1).transpose(1, 2), zone_temp.transpose(1, 2))#.transpose(1, 2)
        semantic_temp = semantic_temp.mean(dim=-1, keepdim=True)

        link_loss = l1 + l2
        ent_loss = e1 + e2

        qs = torch.cat([zone_x, semantic_x], dim=1)

        return link_loss + ent_loss, [zone_x, zone_temp.transpose(1, 2)], [semantic_x, semantic_temp], qs


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, device='cuda:0'):
        super(DomainDiscriminator, self).__init__()

        self.adversarial_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 2)).to(device)

    def forward(self, embed, alpha, if_reverse):
        if if_reverse:
            embed = GradReverse.grad_reverse(embed, alpha)
        else:
            embed = Grad.grad(embed, alpha)

        out = self.adversarial_mlp(embed)

        return F.log_softmax(out, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim=16, device='cuda:0'):
        super(Encoder, self).__init__()
        self.gnn1 = TGCN(hidden_dim)
        self.gnn2 = TGCN(hidden_dim)
        self.gnn3 = TGCN(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim*3+31, hidden_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))

        self.device = device
        self.hidden_dim = hidden_dim

        self.semantic1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fuse_weight = nn.Parameter(torch.randn(1, hidden_dim)).unsqueeze(0).to('cuda:0')

    def forward(self, node_seq, label_seq, t, adj1, adj2, adj3, pre_train=False):
        batch_size = node_seq.shape[0]
        length = node_seq.shape[1]
        node_seq = node_seq.view(batch_size, length, -1)
        node_num = node_seq.shape[2]

        adj1 = torch.from_numpy(adj1).to(self.device).float()
        adj2 = torch.from_numpy(adj2).to(self.device).float()
        adj3 = torch.from_numpy(adj3).to(self.device).float()

        node_embed1 = self.gnn1(node_seq, adj1)
        node_embed2 = self.gnn2(node_seq, adj2)
        node_embed3 = self.gnn3(node_seq, adj3)

        t = t.repeat(1, node_num, 1)
        concat_node = torch.cat([node_embed1, node_embed2, node_embed3, t], dim=-1)
        fused_node = self.mlp(concat_node)

        rec_adj1 = F.sigmoid(torch.bmm(self.semantic1(fused_node), fused_node.transpose(1, 2)))
        rec_adj2 = F.sigmoid(torch.bmm(self.semantic2(fused_node), fused_node.transpose(1, 2)))
        rec_adj3 = F.sigmoid(torch.bmm(self.semantic3(fused_node), fused_node.transpose(1, 2)))
        rec_loss = F.mse_loss(rec_adj1, adj1) + F.mse_loss(rec_adj2, adj2) + F.mse_loss(rec_adj3, adj3)

        weighted_fuse_node = fused_node * self.fuse_weight
        fused_adj = F.softmax(F.relu(torch.bmm(weighted_fuse_node, weighted_fuse_node.transpose(1, 2))), dim=-1)
        ent_loss = (-fused_adj * torch.log(fused_adj + 1e-15)).sum(dim=-1).mean()

        return fused_node, fused_adj, rec_loss+ent_loss


class SFMGTL(nn.Module):
    def __init__(self, hidden_dim=16, device='cuda:0'):
        super(SFMGTL, self).__init__()
        self.s_encoder = Encoder(hidden_dim=hidden_dim, device=device)
        self.t_encoder = Encoder(hidden_dim=hidden_dim, device=device)
        self.discriminator = DomainDiscriminator(hidden_dim)
        self.device = device

        self.s_diffpool = DiffPool(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
        self.t_diffpool = DiffPool(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
        self.s_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.s_head_zone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_dim, 1))
        self.s_head_semantic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 1))
        self.t_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.t_head_zone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_dim, 1))
        self.t_head_semantic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 1))

        # Common Knowledge
        self.common_attention = nn.ModuleList()
        for _ in range(9):
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))

        self.knowledge_number = 3
        self.node_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')
        self.zone_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')
        self.semantic_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')

        self.private_node_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')
        self.private_zone_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')
        self.private_semantic_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to('cuda:0')

        self.temperature = hidden_dim ** 0.5

    def forward(self, s_x, s_y, s_t, s_adj1, s_adj2, s_adj3,
                      t_x, t_y, t_t, t_adj1, t_adj2, t_adj3, alpha, if_reverse):
        s_fused_node, s_fused_adj, s_rec_loss = self.s_encoder(s_x, s_y, s_t, s_adj1, s_adj2, s_adj3)
        t_fused_node, t_fused_adj, t_rec_loss = self.t_encoder(t_x, t_y, t_t, t_adj1, t_adj2, t_adj3)

        s_aux_loss, s_zone_temp, s_semantic_temp, s_qs = self.s_diffpool(s_fused_node, s_fused_adj, s_y.view(s_y.shape[0], 1, -1))
        t_aux_loss, t_zone_temp, t_semantic_temp, t_qs = self.t_diffpool(t_fused_node, t_fused_adj, t_y.view(t_y.shape[0], 1, -1))

        node_q = self.common_attention[0](s_fused_node)
        k = self.common_attention[1](self.node_knowledge)
        v = self.common_attention[2](self.node_knowledge)
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_fused_node = attn @ v

        zone_q = self.common_attention[3](s_zone_temp[0])
        k = self.common_attention[4](self.zone_knowledge)
        v = self.common_attention[5](self.zone_knowledge)
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](s_semantic_temp[0])
        k = self.common_attention[7](self.semantic_knowledge)
        v = self.common_attention[8](self.semantic_knowledge)
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_semantic_temp[0] = attn @ v

        s_pred = self.s_head(s_fused_node)
        s_pred_zone = self.s_head_zone(s_zone_temp[0])
        s_pred_semantic = self.s_head_semantic(s_semantic_temp[0])
        s_zone_loss = F.mse_loss(s_pred_zone, s_zone_temp[1])
        s_semantic_loss = F.mse_loss(s_pred_semantic, s_semantic_temp[1])
        s_aux_loss += s_zone_loss
        s_aux_loss += s_semantic_loss
        s_aux_loss += s_rec_loss

        s_q = torch.cat([node_q, zone_q, semantic_q], dim=1)

        #############################################################################
        node_q = self.common_attention[0](t_fused_node)
        k = self.common_attention[1](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        v = self.common_attention[2](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_fused_node = attn @ v

        zone_q = self.common_attention[3](t_zone_temp[0])
        k = self.common_attention[4](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        v = self.common_attention[5](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](t_semantic_temp[0])
        k = self.common_attention[7](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        v = self.common_attention[8](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_semantic_temp[0] = attn @ v

        t_pred = self.t_head(t_fused_node)
        t_pred_zone = self.t_head_zone(t_zone_temp[0])
        t_pred_semantic = self.t_head_semantic(t_semantic_temp[0])
        t_zone_loss = F.mse_loss(t_pred_zone, t_zone_temp[1])
        t_semantic_loss = F.mse_loss(t_pred_semantic, t_semantic_temp[1])
        t_aux_loss += t_zone_loss
        t_aux_loss += t_semantic_loss
        t_aux_loss += t_rec_loss

        t_q = torch.cat([node_q, zone_q, semantic_q], dim=1)

        # Adversarial
        s_cls = self.discriminator(s_q, alpha, if_reverse).view(-1, 2)
        t_cls = self.discriminator(t_q, alpha, if_reverse).view(-1, 2)
        s_label = torch.zeros(s_cls.shape[0]).long().to(self.device)
        t_label = torch.ones(t_cls.shape[0]).long().to(self.device)

        acc_t = ((torch.exp(t_cls)[:, 1] > 0.5).float().sum()) / (t_cls.shape[0])
        acc_s = ((torch.exp(s_cls)[:, 0] > 0.5).float().sum()) / (s_cls.shape[0])
        accuracy = (acc_s + acc_t) / 2
        adversarial_s_loss = F.nll_loss(s_cls, s_label)
        adversarial_t_loss = F.nll_loss(t_cls, t_label)

        return s_pred, t_pred, s_aux_loss, t_aux_loss, \
               accuracy, (adversarial_s_loss+adversarial_t_loss)

    def evaluation(self, x, y, t, adj1, adj2, adj3):
        fused_node, fused_adj, rec_loss = self.t_encoder(x, y, t, adj1, adj2, adj3)

        aux_loss, zone_temp, semantic_temp, t_qs = self.t_diffpool(fused_node, fused_adj,
                                                                   y.view(y.shape[0], 1, -1))
        node_q = self.common_attention[0](fused_node)
        k = self.common_attention[1](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        v = self.common_attention[2](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        fused_node = attn @ v

        zone_q = self.common_attention[3](zone_temp[0])
        k = self.common_attention[4](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        v = self.common_attention[5](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](semantic_temp[0])
        k = self.common_attention[7](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        v = self.common_attention[8](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        semantic_temp[0] = attn @ v

        pred = self.t_head(fused_node)
        pred_zone = self.t_head_zone(zone_temp[0])
        pred_semantic = self.t_head_semantic(semantic_temp[0])
        zone_loss = F.mse_loss(pred_zone, zone_temp[1])
        semantic_loss = F.mse_loss(pred_semantic, semantic_temp[1])
        aux_loss += zone_loss
        aux_loss += semantic_loss
        aux_loss += rec_loss

        return pred, aux_loss