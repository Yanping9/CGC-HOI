import torch
import torch.nn.functional as F

from torch import nn, Tensor

from transformers import (
    TransformerEncoder,
    HORefineDecoder
)

from ops import (
    compute_spatial_encodings,
    compute_prior_scores,
    compute_sinusoidal_pe,
    match_pose2region
)


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size, num_ho_layers):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)
        self.ho_refine_decoder = HORefineDecoder(num_layers=num_ho_layers)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, h_feas: Tensor, o_feas: Tensor, pose_feas: Tensor,
                h_pe: Tensor, o_pe: Tensor, spa_feas: Tensor) -> Tensor:
        h_feas, o_feas = self.ho_refine_decoder(h_feas, o_feas, pose_feas, h_pe, o_pe)
        ho_feas = torch.cat([h_feas, o_feas], dim=1),
        if isinstance(ho_feas, tuple):
            ho_feas = ho_feas[0]
        ho_feas = self.ln1(self.fc1(ho_feas))
        spa_feas = self.ln2(self.fc2(spa_feas))
        z = F.relu(torch.cat([ho_feas, spa_feas], dim=-1))
        z = self.mlp(z)
        return z

class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, num_ho_layers, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.box_pe_linear = nn.Linear(512, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_linear = nn.Linear(768, 256)
        self.conv_line = nn.Conv2d(2048, 256, kernel_size=1)
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size, num_ho_layers=num_ho_layers)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def transfer_pose_features(self, feat):
        feat = feat.permute(0, 2, 3, 1)
        feat = feat.reshape(-1, 768)
        return self.pose_linear(feat).unsqueeze(1)
    def transfer_line_features(self, feat):
        x = self.pool(feat)  # 输出形状: (2, 2048, 1, 1)
        x = self.conv_line(x)  # 输出形状: (2, 384, 1, 1)
        x = x.view(x.size(0), -1)
        return x

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe

    def compute_hum_obj_pe(self, box_pe_hum, box_pe_obj, line_pe):
        box_pe_hum = self.box_pe_linear(box_pe_hum)
        box_pe_obj = self.box_pe_linear(box_pe_obj)
        line_pe = line_pe.repeat(box_pe_hum.size(0), 1)
        box_pe_hum = box_pe_hum + line_pe
        box_pe_obj = box_pe_obj + line_pe
        return box_pe_hum.unsqueeze(1), box_pe_obj.unsqueeze(1)

    def forward(self, region_props, image_sizes, pose=None, line=None, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        pose_features = []
        line_feas = self.transfer_line_features(line)
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # Enumerate instance pairs
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            x = x.flatten(); y = y.flatten()
            # print(boxes[x_keep])
            # pose_feature, pose_idxs = match_pose2region(boxes[x_keep], pose[i])
            pose_feature, _ = match_pose2region(boxes, labels, pose[i])
            pose_features.append(pose_feature.mean(dim=0))
            pose_feature = self.transfer_pose_features(pose_feature)
            # print(pose_features.shape)
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))
            embeds = embeds.squeeze(1)
            pe_hum, pe_obj = self.compute_hum_obj_pe(box_pe[x_keep], box_pe[y_keep], line_feas[i].unsqueeze(0))
            # Compute human-object queries
            ho_q = self.mmf(
                embeds[x_keep].unsqueeze(1), embeds[y_keep].unsqueeze(1), pose_feature, pe_hum, pe_obj,
                pairwise_spatial_reshaped[x_keep, y_keep]
            )
            # ho_q = self.mmf(
            #     torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
            #     pairwise_spatial_reshaped[x_keep, y_keep]
            # )
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))
            object_types.append(labels[y_keep])
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1),
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds, pose_features