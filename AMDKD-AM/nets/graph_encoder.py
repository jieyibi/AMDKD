import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    #残差连接

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        if isinstance(input, list): # attn loss
            if isinstance(self.module, MultiHeadAttention):
                out1,out2 = self.module(input[0])
                input[0] = out1+input[0]
                if input[1] == 2:
                    input[1] = out2
                else:
                    input[1] = input[1]+1
            else:
                input[0] = self.module(input[0])+input[0]
            return input
        else:
            return input + self.module(input)

class ProbAttention(nn.Module):
    def __init__(self,  factor , head_depth = 16, inf = 1e+10, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.inf = inf
        self.scale = math.sqrt(head_depth)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [H, B, L, D]
        H, B, L_K, E  = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(H, B, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparse measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)# Measurement M = MAX(S-)-MEAN(S-)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(H)[:, None, None],
                     torch.arange(B)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        H, B, L_V, D = V.shape
        # if not self.mask_flag:
        #     # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(H, B, L_Q, V_sum.shape[-1]).clone()
        # else: # use mask
        #     assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
        #     contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, mask):
        H, B, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)[s1]

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        context_in[torch.arange(H)[:, None, None],
                   torch.arange(B)[None, :, None],
                   index, :] = torch.matmul(attn, V) #softmax(QK/sqrt(d))*V [32,8,96,64]
        # if self.output_attention:
        #     attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
        #     attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        #     return (context_in, attns)
        # else:
        return (context_in, None)

    def forward(self, x, mask = None):
        Q, K, V = x
        H, B, L_Q, D = Q.shape # L_Q = L_K = N
        _, _, L_K, _ = K.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        scores_top, index = self._prob_QK(Q, K, sample_k=U_part, n_top=u)

        # add scale factor
        scale = 1/self.scale
        if scale is not None:
            logits = scores_top * scale
        # #Bello的文章里QK*scale之后经过clip*tanh(·),再softmax效果比较好
        # if self.clip is not None:
        #     logits = self.clip * torch.tanh(logits)


        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, B, N, N).expand_as(logits)
            logits[mask] = -np.inf

        # get the context
        context = self._get_initial_context(V, L_Q)
        # update the context with selected top_k queries
        logits, attn = self._update_context(context, V, logits, index, mask)

        return logits

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))#初始化权重
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))


        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf


        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out

class student_Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(student_Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            if isinstance(input,list):
                input[0] = self.normalizer(input[0].view(-1, input[0].size(-1))).view(*input[0].size())
            else:
                return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            if isinstance(input, list):
                input[0] = self.normalizer(input[0].permute(0, 2, 1)).permute(0, 2, 1)
            else:
                return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
        return input
        # if isinstance(self.normalizer, nn.BatchNorm1d):
        #     return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        # elif isinstance(self.normalizer, nn.InstanceNorm1d):
        #     return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        # else:
        #     assert self.normalizer is None, "Unknown normalizer type"
        #     return input, attn

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            if isinstance(input,list):
                input[0] = self.normalizer(input[0].view(-1, input[0].size(-1))).view(*input[0].size())
            else:
                return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            if isinstance(input, list):
                input[0] = self.normalizer(input[0].permute(0, 2, 1)).permute(0, 2, 1)
            else:
                return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
        return input

class student_MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(student_MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            student_Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            student_Normalization(embed_dim, normalization)
        )

class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class student_GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
    ):
        super(student_GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            student_MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))


    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)
        graph_embedding = h.mean(dim=1)

        return (
            h, # (batch_size, graph_size, embed_dim) #if attn loss, [h, attn]
            graph_embedding, # average to get embedding of graph, (batch_size, embed_dim)
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))


    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)
        graph_embedding = h.mean(dim=1)

        return (
            h, # (batch_size, graph_size, embed_dim) #if attn loss, [h, attn]
            graph_embedding, # average to get embedding of graph, (batch_size, embed_dim)
        )
