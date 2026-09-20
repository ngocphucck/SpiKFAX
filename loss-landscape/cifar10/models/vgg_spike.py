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

        # print('Linear shape: ', x.shape)
        # exit()
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

        # print('CNN Shape: ', x.shape)
        # exit()
        
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
        # print('Pooling shape: ', x.shape)
        x_flat = x.contiguous().view(B * T, C, H, W)
        out = self.pool(x_flat)
        return out.view(B, T, C, out.shape[2], out.shape[3])


class SpikingVGG11(nn.Module):
    def __init__(self):
        super().__init__()

        beta = 0.9

        self.features = nn.Sequential(
            SpikeConv2d(3, 64, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(64, 128, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(128, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(256, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(256, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
        )
        
        # Assumes CIFAR-10 32x32 input -> pooled 5 times -> 1x1 spatial size
        self.classifier = nn.Sequential(
            SpikeLinear(512 * 1 * 1, 4096, beta=beta),
            SpikeLinear(4096, 4096, beta=beta),
            SpikeLinear(4096, 10, beta=beta)
        )

    def forward(self, x):
        # print('x shape: ', x.shape)
        # exit()
        x = x.transpose(0, 1)
        # print(x.shape)
        # print('Data')
        # print(x.shape)
        # print('Min: ', torch.min(x))
        # print('Max: ', torch.max(x))
        # exit()

        x = self.features(x)

        # print('After CNN')
        # print(x.shape)
        # print('Min: ', torch.min(x))
        # print('Max: ', torch.max(x))
        # print('Shape x: ', x.shape)
        # exit()
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)

        # print('Out: ', self.classifier(x).shape)
        # exit()
        # a = torch.sum(self.classifier(x), dim=1)
        # print('After Linear!')
        # print(a.shape)
        # print('Min: ', torch.min(a))
        # print('Max: ', torch.max(a))
        # exit()
        
        return self.classifier(x)

def VGGSpike():
    
    return SpikingVGG11()