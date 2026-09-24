import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.optim import Optimizer


class SpiKFAX(Optimizer):
    """
    K-FAC-style natural gradient optimizer for SpikeLinear and SpikeConv2d layers,
    with explicit handling for BatchNorm2d scale/shift parameters and biases.
    """

    def __init__(
        self,
        model_or_layers,
        lr=1e-2,
        damping=1e-2,
        bias_lr=None,
        eps=1e-8,
        stat_decay=0.95,
        update_freq=1,
        max_update_ratio=None,
        momentum=0.0,
        fisher_order="diag",
    ):
        if bias_lr is None:
            bias_lr = lr
        if fisher_order not in ("diag", "2term"):
            raise ValueError("fisher_order must be 'diag' or '2term'")

        self.layers = self._discover_layers(model_or_layers)
        self._param_to_layer = {}
        self._param_kind = {}
        self._step_count = 0
        self._warned_conv_2term = False

        params = []
        
        # Collect parameters from discovered spike layers (weights and biases)
        for layer in self.layers:
            kind = self._layer_kind(layer)
            sub = layer.linear if kind == "linear" else layer.conv
            params.append(sub.weight)
            self._param_to_layer[sub.weight] = layer
            self._param_kind[sub.weight] = kind
            if sub.bias is not None:
                params.append(sub.bias)

        # Also discover and append any standalone parameters not bound to Spike layers 
        # (e.g., BatchNorm2d gamma/beta parameters, running stats are ignored since they don't have grads)
        if isinstance(model_or_layers, nn.Module):
            spike_module_set = set(self.layers)
            for m in model_or_layers.modules():
                # If it's a BatchNorm2d or any other module not part of SpikeLinear/SpikeConv2d
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        if p.requires_grad and p not in self._param_to_layer:
                            params.append(p)

        
        defaults = dict(
            lr=lr,
            damping=damping,
            bias_lr=bias_lr,
            eps=eps,
            stat_decay=stat_decay,
            update_freq=update_freq,
            max_update_ratio=max_update_ratio,
            momentum=momentum,
            fisher_order=fisher_order,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _layer_kind(m):
        has_common = hasattr(m, "get_membrane_gradients") and hasattr(m, "inputs")
        if not has_common:
            return None
        if hasattr(m, "linear") and isinstance(m.linear, nn.Linear):
            return "linear"
        if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
            return "conv"
        return None

    @classmethod
    def _is_spike_linear(cls, m):
        return cls._layer_kind(m) is not None

    @classmethod
    def _discover_layers(cls, obj):
        if isinstance(obj, (list, tuple)):
            layers = [m for m in obj if cls._is_spike_linear(m)]
            if not layers:
                raise TypeError("None of the provided modules look like SpikeLinear or SpikeConv2d layers")
            return layers

        if isinstance(obj, nn.Module):
            layers = [m for m in obj.modules() if cls._is_spike_linear(m)]
            if not layers:
                raise TypeError(
                    "No SpikeLinear-like or SpikeConv2d-like submodules found in "
                    "the given model."
                )
            return layers

        raise TypeError(
            "SpikeFAC needs the model itself or a list of SpikeLinear/SpikeConv2d "
            "layers, not model.parameters()."
        )

    @staticmethod
    def _factored_damping(A, G, damping, eps):
 
        m, n = A.shape[0], G.shape[0]
        trace_a = torch.trace(A).clamp(min=1e-12) / m
        trace_g = torch.trace(G).clamp(min=1e-12) / n
        pi = torch.sqrt(trace_a / trace_g)

        sqrt_term = damping ** 0.5
        A_d = (
            A
            + pi * sqrt_term * torch.eye(m, device=A.device, dtype=A.dtype)
            + eps * torch.eye(m, device=A.device, dtype=A.dtype)
        )
        G_d = (
            G
            + (1.0 / pi) * sqrt_term * torch.eye(n, device=G.device, dtype=G.dtype)
            + eps * torch.eye(n, device=G.device, dtype=G.dtype)
        )
        return A_d, G_d

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            bias_lr = group["bias_lr"]
            damping = group["damping"]
            eps = group["eps"]
            stat_decay = group["stat_decay"]
            update_freq = max(1, group["update_freq"])
            max_update_ratio = group["max_update_ratio"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                layer = self._param_to_layer.get(p, None)
                state = self.state[p]

                # Unmapped parameters (Biases, BatchNorm2d weights/biases, etc.):
                # Standard gradient step with momentum fallback.
                if layer is None:
                    # continue
                    # print('MLP Layers: ', p.shape)
                    if momentum > 0.0:
                        buf = state.get("aux_momentum_buf")
                        if buf is None:
                            buf = p.grad.clone()
                        else:
                            buf.mul_(momentum).add_(p.grad)
                        state["aux_momentum_buf"] = buf
                        p.add_(buf, alpha=-bias_lr)
                    else:
                        p.add_(p.grad, alpha=-bias_lr)
                    continue

                fisher_order = group["fisher_order"]
                kind = self._param_kind[p]

                if kind == "conv":
                    # continue
                    # print('CNN Layers: ', p.shape)

                    if fisher_order == "2term" and not self._warned_conv_2term:
                        print(
                            "[SpikeFAC] fisher_order='2term' is only implemented for "
                            "SpikeLinear layers; SpikeConv2d layers will use 'diag' "
                            "instead (this warning prints once)."
                        )
                        self._warned_conv_2term = True

                    refresh = (state.get("A_inv") is None) or (self._step_count % update_freq == 0)

                    if refresh:
                        A_batch, G_batch = self._compute_fisher_batch_stats_conv(layer)
                        if A_batch is None:
                            if state.get("A_inv") is None:
                                continue
                        else:
                            if "A_ema" not in state or stat_decay <= 0.0:
                                state["A_ema"] = A_batch
                                state["G_ema"] = G_batch
                            else:
                                state["A_ema"].mul_(stat_decay).add_(A_batch, alpha=1 - stat_decay)
                                state["G_ema"].mul_(stat_decay).add_(G_batch, alpha=1 - stat_decay)

                            # --- Remark 1: factored, pi-rescaled damping (was: naive shared (damping+eps)*I) ---
                            A_d, G_d = self._factored_damping(state["A_ema"], state["G_ema"], damping, eps)
                            state["A_inv"] = torch.linalg.inv(A_d)
                            state["G_inv"] = torch.linalg.inv(G_d)

                    A_inv = state.get("A_inv")
                    G_inv = state.get("G_inv")
                    if A_inv is None:
                        continue

                    weight_shape = p.shape
                    grad_mat = p.grad.reshape(weight_shape[0], -1)
                    nat_grad_mat = G_inv @ grad_mat @ A_inv
                    nat_grad = nat_grad_mat.reshape(weight_shape)

                elif fisher_order == "2term":
                    # continue
                    print('Layers: ', p.shape)
                    nat_grad = self._natural_grad_2term(
                        layer, p.grad, state, damping, eps, stat_decay, update_freq
                    )
                    if nat_grad is None:
                        continue
                else:
                    # continue
                    # print('Layer: ', p.shape)
                    refresh = (state.get("A_inv") is None) or (self._step_count % update_freq == 0)

                    if refresh:
                        A_batch, G_batch = self._compute_fisher_batch_stats(layer)
                        if A_batch is None:
                            if state.get("A_inv") is None:
                                continue
                        else:
                            if "A_ema" not in state or stat_decay <= 0.0:
                                state["A_ema"] = A_batch
                                state["G_ema"] = G_batch
                            else:
                                state["A_ema"].mul_(stat_decay).add_(A_batch, alpha=1 - stat_decay)
                                state["G_ema"].mul_(stat_decay).add_(G_batch, alpha=1 - stat_decay)

                            # --- Remark 1: factored, pi-rescaled damping (was: naive shared (damping+eps)*I) ---
                            A_d, G_d = self._factored_damping(state["A_ema"], state["G_ema"], damping, eps)
                            state["A_inv"] = torch.linalg.inv(A_d)
                            state["G_inv"] = torch.linalg.inv(G_d)

                    A_inv = state.get("A_inv")
                    G_inv = state.get("G_inv")
                    if A_inv is None:
                        continue

                    grad = p.grad
                    nat_grad = G_inv @ grad @ A_inv

                grad = p.grad
                if max_update_ratio is not None:
                    grad_norm = grad.norm()
                    nat_norm = nat_grad.norm()
                    cap = max_update_ratio * grad_norm + eps
                    if nat_norm > cap:
                        nat_grad = nat_grad * (cap / (nat_norm + eps))

                if momentum > 0.0:
                    buf = state.get("weight_momentum_buf")
                    if buf is None:
                        buf = nat_grad.clone()
                    else:
                        buf.mul_(momentum).add_(nat_grad)
                    state["weight_momentum_buf"] = buf
                    p.add_(buf, alpha=-lr)
                else:
                    p.add_(nat_grad, alpha=-lr)
        # exit()
        return loss

    @staticmethod
    def _compute_fisher_batch_stats_conv(layer):
        inputs = getattr(layer, "inputs", None)
        if not inputs:
            raise AttributeError("SpikeConv2d has no cached `.inputs`.")
        mem_grads = layer.get_membrane_gradients()

        valid_x, valid_d = [], []
        for x_t, d_t in zip(inputs, mem_grads):
            if d_t is not None:
                valid_x.append(x_t)
                valid_d.append(d_t)

        if not valid_x:
            return None, None

        conv = layer.conv
        if conv.groups != 1:
            raise NotImplementedError("SpikeFAC conv support assumes groups=1.")

        patches_per_t = [
            Fnn.unfold(
                x_t,
                kernel_size=conv.kernel_size,
                dilation=conv.dilation,
                padding=conv.padding,
                stride=conv.stride,
            )
            for x_t in valid_x
        ]
        P = torch.stack(patches_per_t, dim=0)
        _, B, K, _ = P.shape
        a_bar = P.mean(dim=(0, 3))
        A = (a_bar.t() @ a_bar) / B

        D = torch.stack(valid_d, dim=0)
        delta_sum = D.sum(dim=(0, 3, 4))
        G = (delta_sum.t() @ delta_sum) / B

        return A, G

    @staticmethod
    def _compute_fisher_batch_stats(layer):
        inputs = getattr(layer, "inputs", None)
        if not inputs:
            raise AttributeError("SpikeLinear has no cached `.inputs`.")
        mem_grads = layer.get_membrane_gradients()

        valid_x, valid_d = [], []
        for x_t, d_t in zip(inputs, mem_grads):
            if d_t is not None:
                valid_x.append(x_t)
                valid_d.append(d_t)

        if not valid_x:
            return None, None

        X = torch.stack(valid_x, dim=0)
        T, B, in_dim = X.shape
        x_mean = X.mean(dim=0)
        A = (x_mean.t() @ x_mean) / B

        D = torch.stack(valid_d, dim=0)
        delta_sum = D.sum(dim=0)
        G = (delta_sum.t() @ delta_sum) / B

        return A, G

    @staticmethod
    def _compute_2term_batch_stats(layer):
        inputs = getattr(layer, "inputs", None)
        if not inputs:
            raise AttributeError("SpikeLinear has no cached `.inputs`.")
        mem_grads = layer.get_membrane_gradients()

        valid_x, valid_d = [], []
        for x_t, d_t in zip(inputs, mem_grads):
            if d_t is not None:
                valid_x.append(x_t)
                valid_d.append(d_t)

        if not valid_x:
            return None, None, None, None

        X = torch.stack(valid_x, dim=0)
        T, B, in_dim = X.shape

        x_bar = X.mean(dim=0)
        diff = X - x_bar.unsqueeze(0)
        diff_flat = diff.reshape(T * B, in_dim)
        Sigma_x = (diff_flat.t() @ diff_flat) / (T * B)

        D = torch.stack(valid_d, dim=0)
        delta_sum = D.sum(dim=0)
        G = (delta_sum.t() @ delta_sum) / B

        D_flat = D.reshape(T * B, -1)
        G_diag = (D_flat.t() @ D_flat) / B

        return x_bar, Sigma_x, G, G_diag

    def _natural_grad_2term(self, layer, grad, state, damping, eps, stat_decay, update_freq):
        refresh = (state.get("woodbury_ready") is None) or (self._step_count % update_freq == 0)

        if refresh:
            x_bar, Sigma_x, G, G_diag = self._compute_2term_batch_stats(layer)
            if x_bar is None:
                if state.get("woodbury_ready") is None:
                    return None
            else:
                if state.get("woodbury_ready") is None or stat_decay <= 0.0:
                    state["x_bar_ema"] = x_bar
                    state["Sigma_x_ema"] = Sigma_x
                    state["G_ema2"] = G
                    state["G_diag_ema"] = G_diag
                else:
                    state["x_bar_ema"] = x_bar
                    state["Sigma_x_ema"].mul_(stat_decay).add_(Sigma_x, alpha=1 - stat_decay)
                    state["G_ema2"].mul_(stat_decay).add_(G, alpha=1 - stat_decay)
                    state["G_diag_ema"].mul_(stat_decay).add_(G_diag, alpha=1 - stat_decay)

                dev, dt = grad.device, grad.dtype

                Sigma_x_d, G_diag_d = self._factored_damping(state["Sigma_x_ema"], state["G_diag_ema"], damping, eps)
                out_dim = state["G_ema2"].shape[0]
                G_d = state["G_ema2"] + (damping + eps) * torch.eye(out_dim, device=dev, dtype=dt)

                V = state["x_bar_ema"].t()

                Sigma_x_d_inv = torch.linalg.inv(Sigma_x_d)
                G_diag_d_inv = torch.linalg.inv(G_diag_d)
                G_d_half = torch.linalg.cholesky(G_d)

                P = V.t() @ Sigma_x_d_inv @ V
                Q = G_d_half.t() @ G_diag_d_inv @ G_d_half

                Lam_P, E_P = torch.linalg.eigh(P)
                Lam_Q, E_Q = torch.linalg.eigh(Q)

                state["Sigma_x_d_inv"] = Sigma_x_d_inv
                state["G_diag_d_inv"] = G_diag_d_inv
                state["G_d_half"] = G_d_half
                state["V"] = V
                state["Lam_P"], state["E_P"] = Lam_P, E_P
                state["Lam_Q"], state["E_Q"] = Lam_Q, E_Q
                state["B_val"] = V.shape[1]
                state["woodbury_ready"] = True

        if state.get("woodbury_ready") is None:
            return None

        Sigma_x_d_inv = state["Sigma_x_d_inv"]
        G_diag_d_inv = state["G_diag_d_inv"]
        G_d_half = state["G_d_half"]
        V = state["V"]
        Lam_P, E_P = state["Lam_P"], state["E_P"]
        Lam_Q, E_Q = state["Lam_Q"], state["E_Q"]
        B_val = state["B_val"]

        w = G_diag_d_inv @ grad @ Sigma_x_d_inv
        z_mat = G_d_half.t() @ w @ V
        z_tilde = E_Q.t() @ z_mat @ E_P
        denom = B_val + Lam_Q.unsqueeze(1) * Lam_P.unsqueeze(0)
        y_tilde = z_tilde / denom
        y_mat = E_Q @ y_tilde @ E_P.t()
        correction_raw = G_d_half @ y_mat @ V.t()
        correction = G_diag_d_inv @ correction_raw @ Sigma_x_d_inv

        return w - correction

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)