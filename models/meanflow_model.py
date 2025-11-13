import torch
from torch import nn

from models.node_feature_net import MeanFlowNodeFeatureNet, MeanFlowNodeFeatureNetv2
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du


from data.so3_utils import calc_rot_vf, exp


class MeanFlowModel(nn.Module):

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )
        if model_conf.node_features.get("version", "v1") == "v2":
            self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        else:
            self.node_feature_net = MeanFlowNodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s
            )
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

            if self._model_conf.get("strict_time_conditioning", False):
                self.trunk[f"skip_embed_{b}"] = nn.Linear(
                    self._model_conf.node_embed_size, self._ipa_conf.c_s
                )

    def forward(self, trans_t, rotmat_t, t, r, feats):
        node_mask = feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = feats["diffuse_mask"]
        res_index = feats["res_idx"]

        so3_t = t
        r3_t = t
        r = r
        trans_t = trans_t
        rotmats_t = rotmat_t

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t, r3_t, r, node_mask, diffuse_mask, res_index
        )

        trans_sc = torch.zeros_like(trans_t)

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            if self._model_conf.get("strict_time_conditioning", False):
                node_embed = node_embed + self.trunk[f"skip_embed_{b}"](init_node_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None]
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats

    def set_interpolant(self, interpolant):
        self.interpolant = interpolant

    def avg_vel(self, trans_t, rotmat_t, t, r, feats, return_model_output=False):
        trans_1, rotmat_1 = self(trans_t, rotmat_t, t, r, feats)

        if self._model_conf.get("cond_vel_parameterization", True):
            # trans_vf = (trans_1 - trans_t) / ((1 - t)[..., None] + 1e-6)
            trans_vf = (trans_1 - trans_t) / torch.clamp(1 - t, min=1e-1)[..., None]
            rot_vf = self.interpolant.rots_cond_vf(t, rotmat_t, rotmat_1)
        else:
            trans_vf = trans_1 - trans_t
            rot_vf = calc_rot_vf(rotmat_t, rotmat_1)

        if return_model_output:
            return trans_vf, rot_vf, (trans_1, rotmat_1)
        else:
            return trans_vf, rot_vf

    def forward_flow(self, trans_t, rotmat_t, t, r, feats):
        trans_vf, rot_vf = self.avg_vel(trans_t, rotmat_t, t, r, feats)

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def inference_avg_vel(self, trans_t, rotmat_t, t, r, feats):
        trans_1, rotmat_1 = self(trans_t, rotmat_t, t, r, feats)
        trans_vf = (trans_1 - trans_t) / torch.clamp(1 - t, min=1e-1)[..., None]
        rot_vf = calc_rot_vf(rotmat_t, rotmat_1)
        rot_vf *= self.interpolant._rots_cfg.exp_rate
        return trans_vf, rot_vf

    def inference_forward_flow(self, trans_t, rotmat_t, t, r, feats):
        trans_vf, rot_vf = self.inference_avg_vel(trans_t, rotmat_t, t, r, feats)

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def jvp_avg_vel(self, trans_t, rotmat_t, t, r, feats, tangent):
        """
        tangent: (d_trans, d_rot, d_t, d_r)
        """

        def u(trans_t, rotmat_t, t, r):
            return self.avg_vel(trans_t, rotmat_t, t, r, feats)

        return torch.func.jvp(u, (trans_t, rotmat_t, t, r), tangent)


def project_to_SO3(rotmats: torch.Tensor) -> torch.Tensor:
    """
    Project a tensor to the SO(3) manifold.
    Shape:
        rotmats: (B, 3, 3)
    """
    # Batched SVD-based projection to the nearest rotation matrix (Frobenius norm)
    # Ensures det(R) = 1 by correcting reflections when needed.
    assert (
        rotmats.dim() == 3 and rotmats.size(-2) == 3 and rotmats.size(-1) == 3
    ), "rotmats must be (B, 3, 3)"

    # U @ Vh gives the orthogonal factor from polar decomposition
    U, S, Vh = torch.linalg.svd(rotmats)
    R = U @ Vh

    # If det(R) < 0, flip the last axis to enforce det = 1
    det_R = torch.det(R)
    sign = torch.where(det_R >= 0, torch.ones_like(det_R), -torch.ones_like(det_R))
    D = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )

    R_corrected = U @ D @ Vh
    return R_corrected


from data.so3_interpolant import SO3Interpolant


class ToySO3MeanFlowModel(nn.Module):

    def __init__(self, cfg):
        super(ToySO3MeanFlowModel, self).__init__()
        self.cfg = cfg

        self.interpolant = None
        self.num_blocks = cfg.num_blocks
        self.emb_dim = cfg.emb_dim

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(9 + 2, self.emb_dim),
                nn.SiLU(),
            )
        )
        for _ in range(self.num_blocks - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.emb_dim + 2, self.emb_dim),
                    nn.SiLU(),
                )
            )

        self.predict_head = nn.Linear(self.emb_dim, 9)

    def forward(self, Rt, t, r):
        """
        Shape:
            Rt: (B, 3, 3)
            t: (B,)
            r: (B,)
        """
        x = Rt.reshape(-1, 9)
        for layer in self.layers:
            x = torch.cat([x, t[..., None], r[..., None]], dim=-1)
            x = layer(x)
        x = self.predict_head(x).view(Rt.shape)
        return project_to_SO3(x)

    def set_interpolant(self, interpolant: SO3Interpolant):
        self.interpolant = interpolant

    def avg_vel(self, Rt, t, r):
        R1 = self(Rt, t, r)

        if self.cfg.get("cond_vel_parameterization", True):
            rot_vf = self.interpolant.cond_vf(t, Rt, R1)
        else:
            rot_vf = calc_rot_vf(Rt, R1)

        return rot_vf

    def forward_flow(self, Rt, t, r):
        rot_vf = self.avg_vel(Rt, t, r)
        Rr = exp(Rt, (r - t)[..., None] * rot_vf)
        return Rr

    def jvp_avg_vel(self, Rt, t, r, tangent):
        """
        tangent: (d_R, d_t, d_r)
        """
        return torch.func.jvp(self.avg_vel, (Rt, t, r), tangent)

    @torch.no_grad()
    def sample(self, x0, num_timesteps=100, verbose=False, return_traj=False):
        ts = torch.linspace(0.01, 1.0, num_timesteps)
        t1 = ts[0]
        traj = [x0]
        num_batch = x0.shape[0]
        device = x0.device

        for i, t2 in enumerate(ts[1:]):
            if verbose:  # and i % 1 == 0:
                print(f"{i=}, t={t1.item():.2f}")

            xt = traj[-1]
            r = t = torch.ones((num_batch,), device=device) * t1
            d_t = t2 - t1

            v_rot = self.avg_vel(xt, t, r)

            # Take reverse step
            xt = self.interpolant.euler_step(d_t, xt, v_rot)

            traj.append(xt)
            t1 = t2

        if return_traj:
            return traj
        else:
            return traj[-1]

    def one_step_sample(self, x0):
        t = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.float32)
        r = torch.ones((x0.shape[0],), device=x0.device, dtype=torch.float32)
        return self.forward_flow(x0, t, r)
