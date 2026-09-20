from torch import nn
import snntorch as snn
import torch


class SpikeLinear(nn.Module):
    def __init__(self, input_dim, output_dim, beta=0.9):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.lif = snn.Leaky(beta=beta)

        # Saved during forward()
        self.spikes = []
        self.membranes = []
        self.inputs = []

    def forward(self, x):
        """
        x: [batch, time, input_dim]

        returns:
            spikes: [batch, time, hidden_dim]
        """

        batch_size, time_steps, _ = x.shape

        # Initial membrane state
        mem = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype
        )

        # Clear previous forward pass
        self.spikes = []
        self.membranes = []
        self.inputs = []

        for t in range(time_steps):
            self.inputs.append(x[:, t, :].detach())

            # Linear
            current = self.linear(x[:, t, :])

            # LIF
            spike, mem = self.lif(current, mem)

            # Keep gradients of timestep states
            if spike.requires_grad:
                spike.retain_grad()

            if mem.requires_grad:
                mem.retain_grad()

            self.spikes.append(spike)
            self.membranes.append(mem)

        # [batch, time, hidden_dim]
        spikes = torch.stack(self.spikes, dim=1)

        return spikes

    def get_spike_gradients(self):
        """
        Returns:
            [T] tensors, where gradients[t] = dL/dSpike_t

        Call after loss.backward().
        """

        return [
            spike.grad.detach().clone()
            if spike.grad is not None else None
            for spike in self.spikes
        ]

    def get_membrane_gradients(self):
        """
        Returns:
            [T] tensors, where gradients[t] = dL/dMem_t

        Call after loss.backward().
        """

        return [
            mem.grad.detach().clone()
            if mem.grad is not None else None
            for mem in self.membranes
        ]

    def get_spike_gradient_norms(self):
        """
        Returns:
            [T] gradient norms for spikes.
        """

        return [
            spike.grad.norm().item()
            if spike.grad is not None else None
            for spike in self.spikes
        ]

    def get_membrane_gradient_norms(self):
        """
        Returns:
            [T] gradient norms for membrane states.
        """

        return [
            mem.grad.norm().item()
            if mem.grad is not None else None
            for mem in self.membranes
        ]


class SpikeConv2d(nn.Module):
    """Convolutional LIF layer with integrated pre-threshold Batch Normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, beta=0.9, use_bn=False):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn
        )
        
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            
        self.lif = snn.Leaky(beta=beta)

        # Caching lists for SpikeFAC
        self.spikes = []
        self.membranes = []
        self.inputs = []

    def forward(self, x):
        """x shape expected: [Batch, Time, Channels, Height, Width]"""
        batch_size, time_steps, C, H, W = x.shape
        
        self.spikes = []
        self.membranes = []
        self.inputs = []
        mem = None
        
        for t in range(time_steps):
            x_t = x[:, t, ...]
            self.inputs.append(x_t.detach())
            
            current = self.conv(x_t)
            
            # Normalize pre-synaptic current before thresholding
            if self.use_bn:
                current = self.bn(current)
                
            if mem is None:
                mem = torch.zeros_like(current)
                
            spike, mem = self.lif(current, mem)
            
            if spike.requires_grad: 
                spike.retain_grad()
            if mem.requires_grad: 
                mem.retain_grad()
                
            self.spikes.append(spike)
            self.membranes.append(mem)
            
        return torch.stack(self.spikes, dim=1)

    def get_membrane_gradients(self):
        return [
            m.grad.detach().clone() if m.grad is not None else None 
            for m in self.membranes
        ]


class SpikeMaxPool2d(nn.Module):
    """Applies MaxPool2d across the temporal dimension seamlessly."""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.contiguous().view(B * T, C, H, W)
        out = self.pool(x_flat)
        return out.view(B, T, C, out.shape[2], out.shape[3])
