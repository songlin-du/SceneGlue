import torch
from torch import nn
from timm.models.layers import trunc_normal_
from .blocks import AttnBlock_ln, Wave_position_encoder, normalize_keypoints, log_optimal_transport, arange_like, MLP, group_decoder_sf


class TransLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x.transpose(1,2)).transpose(1,2)


class GroupingLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 num_group_token,
                 group_projector,
                 mlp_ratio=2.,
                 attn_drop=0.,):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_group_token = num_group_token

        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock_ln(
                    num_heads=num_heads,
                    d_model=dim,
                    attention_dropout=attn_drop,
                    mlp_ratio=mlp_ratio,
                    act=nn.GELU,))
        self.blocks = nn.ModuleList(blocks)
        self.group_projector = group_projector
        self.group_token = nn.Parameter(torch.zeros(1, dim, 2))
        trunc_normal_(self.group_token, std=0.2)

    @property
    def with_group_token(self):
        return True

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :, :-self.num_group_token], x[:, :, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=2)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        b, c, _ = x.shape
        group_token_x, group_token_y = self.group_token.repeat(b, 1, 1), self.group_token.repeat(b, 1, 1)

        x, y = self.concat_x(x, group_token_x), self.concat_x(y, group_token_y)

        for blk in self.blocks:
            x, y = blk(x, y)

        x, group_token_x = self.split_x(x)
        y, group_token_y = self.split_x(y)

        return x, y, group_token_x, group_token_y


class AttentionalGNN(nn.Module):
    def __init__(self,
                 mlp_ratio,
                 embed_dim=[256, 256, 256],
                 depths=[3, 3, 3],
                 num_heads=[4, 4, 4],
                 num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8],
                 attention_dropout=0.,
                 ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups

        i_layer = 0
        dim = int(embed_dim[i_layer])
        self.PF = GroupingLayer(
            dim=dim,
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            num_group_token=num_group_tokens[i_layer],
            group_projector=None,
            mlp_ratio=self.mlp_ratio,
            attn_drop=attention_dropout,
        )

        self.grouping = group_decoder_sf(
            dim=dim,
            num_heads=num_heads[i_layer],)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, desc0, desc1):
        desc0, desc1, group_token_0, group_token_1 = self.PF(desc0, desc1)
        group0 = self.grouping(group_token_0, desc0)
        group1 = self.grouping(group_token_1, desc1)
        return desc0, desc1, group0, group1


def split_desc(desc, N):
    desc_0, desc_1 = desc[:, :, :N], desc[:, :, N:]
    return desc_0, desc_1


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)


class SceneGlue(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'x_ratio': 4,
        'm_ratio': 4,
        'mlp_ratio': 2,
        'embed_dim': [256, 256],
        'depths': [9],
        'num_heads': [4],
        'num_group_tokens': [2, 0],
        'num_output_groups': [2],
        'sinkhorn_iterations': 40,
        'match_threshold': 0.2,
        'attention_dropout': 0.,
        'hard_assignment': True,
        'hard_score': False,
        'keypoint_encoder': [32, 64, 128, 256],
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.wave_kenc = Wave_position_encoder(self.config['descriptor_dim'], self.config['x_ratio'], self.config['m_ratio'], act=nn.ReLU)

        self.gnn = AttentionalGNN(
            self.config['mlp_ratio'],
            embed_dim=self.config['embed_dim'],
            depths=self.config['depths'],
            num_heads=self.config['num_heads'],
            num_group_tokens=self.config['num_group_tokens'],
            num_output_groups=self.config['num_output_groups'],
            attention_dropout=self.config['attention_dropout'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.register_parameter('bin_score', bin_score)

    def forward(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        bs, C, N = desc0.shape
        M = desc1.shape[2]
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        if len(data['scores0'].shape)==2:
            data['scores0'], data['scores1'] = data['scores0'].unsqueeze(1), data['scores1'].unsqueeze(1)
        scores0, scores1 = data['scores0'], data['scores1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        kpts0_o, kpts1_o = kpts0, kpts1
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[2:])
        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        kpts0, kpts1 = torch.cat([kpts0, scores0], dim=1), torch.cat([kpts1, scores1], dim=1)
        desc0, desc1 = self.wave_kenc(desc0, kpts0), self.wave_kenc(desc1, kpts1)
        # desc0, desc1 = desc0 + self.kenc(kpts0), desc1 + self.kenc(kpts1)

        # Multi-layer Transformer network.
        desc0, desc1, group0, group1 = self.gnn(desc0, desc1)

        # viz_TSNE([desc0, desc1], 'sig_a.png')

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1) / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        matches, conf = indices0, mscores0
        valid = matches > -1
        mconf = torch.cat([conf[b, valid[b]] for b in range(conf.size(0))], 0)
        mkpts0 = torch.cat([kpts0_o[b, valid[b], :] for b in range(kpts0_o.size(0))], 0)
        mkpts1 = torch.cat([kpts1_o[b, matches[b, valid[b]], :] for b in range(kpts1_o.size(0))], 0)

        num_matches = valid.sum(1)
        m_bids = []
        for b in range(num_matches.size(0)):
            m_bids += num_matches[b] * [b]
        m_bids = torch.tensor(m_bids, dtype=torch.int64)
        data.update({
            'desc0': desc0,
            'desc1': desc1,
            'mconf': mconf,
            'mkpts0_f': mkpts0,  # use -1 for invalid match
            'mkpts1_f': mkpts1,  # use -1 for invalid match
            'm_bids': m_bids,
            'conf_matrix_with_bin': scores.exp(),
            'attn0': group0,
            'attn1': group1,
        })