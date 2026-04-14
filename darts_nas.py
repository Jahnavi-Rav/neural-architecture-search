import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class DARTSCell(nn.Module):
    """Differentiable Architecture Search Cell"""
    
    PRIMITIVES = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]
    
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        
        # Architecture parameters (alpha)
        self.num_ops = len(self.PRIMITIVES)
        self.alphas = nn.Parameter(torch.randn(self.num_ops))
        
        # Build operations
        self.ops = nn.ModuleList()
        for primitive in self.PRIMITIVES:
            op = self._build_op(primitive, C_in, C_out, stride)
            self.ops.append(op)
    
    def _build_op(self, primitive, C_in, C_out, stride):
        """Build operation based on primitive type"""
        if primitive == 'none':
            return Zero(stride)
        elif primitive == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride=stride, padding=1)
        elif primitive == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride=stride, padding=1)
        elif primitive == 'skip_connect':
            return Identity() if stride == 1 else FactorizedReduce(C_in, C_out)
        elif primitive == 'sep_conv_3x3':
            return SepConv(C_in, C_out, 3, stride)
        elif primitive == 'sep_conv_5x5':
            return SepConv(C_in, C_out, 5, stride)
        elif primitive == 'dil_conv_3x3':
            return DilConv(C_in, C_out, 3, stride, 2)
        elif primitive == 'dil_conv_5x5':
            return DilConv(C_in, C_out, 5, stride, 2)
        else:
            raise ValueError(f"Unknown operation: {primitive}")
    
    def forward(self, x):
        """Forward pass with differentiable architecture"""
        # Apply softmax to architecture parameters
        weights = F.softmax(self.alphas, dim=0)
        
        # Weighted sum of all operations
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output
    
    def get_architecture(self):
        """Get the most likely architecture"""
        weights = F.softmax(self.alphas, dim=0)
        best_op_idx = weights.argmax().item()
        return self.PRIMITIVES[best_op_idx]

class SepConv(nn.Module):
    """Separable Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, padding=0),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, dilation):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, 
                     dilation=dilation, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, padding=0),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(C_out)
    
    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

class DARTSNetwork(nn.Module):
    """Complete DARTS network for image classification"""
    
    def __init__(self, C=16, num_classes=10, layers=8):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        
        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        # Stacked cells
        self.cells = nn.ModuleList()
        C_curr = C
        
        for i in range(layers):
            # Reduction cell at 1/3 and 2/3 of total depth
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                cell = DARTSCell(C_curr // 2, C_curr, stride=2)
            else:
                cell = DARTSCell(C_curr, C_curr, stride=1)
            
            self.cells.append(cell)
        
        # Classification head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_curr, num_classes)
    
    def forward(self, x):
        s = self.stem(x)
        
        for cell in self.cells:
            s = cell(s)
        
        out = self.global_pooling(s)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def get_architecture(self):
        """Extract discovered architecture"""
        architecture = []
        for i, cell in enumerate(self.cells):
            op = cell.get_architecture()
            architecture.append(f"Layer {i}: {op}")
        return architecture

class DARTSSearcher:
    """DARTS architecture search controller"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Separate optimizers for network weights and architecture parameters
        self.w_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )
        
        self.alpha_optimizer = torch.optim.Adam(
            [cell.alphas for cell in self.model.cells],
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
    
    def train_step(self, train_data, val_data):
        """Single training step with bilevel optimization"""
        self.model.train()
        
        # Unpack data
        (train_X, train_y) = train_data
        (val_X, val_y) = val_data
        
        train_X, train_y = train_X.to(self.device), train_y.to(self.device)
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)
        
        # Step 1: Update architecture parameters on validation set
        self.alpha_optimizer.zero_grad()
        logits = self.model(val_X)
        arch_loss = F.cross_entropy(logits, val_y)
        arch_loss.backward()
        self.alpha_optimizer.step()
        
        # Step 2: Update network weights on training set
        self.w_optimizer.zero_grad()
        logits = self.model(train_X)
        weight_loss = F.cross_entropy(logits, train_y)
        weight_loss.backward()
        self.w_optimizer.step()
        
        return weight_loss.item(), arch_loss.item()
    
    def search(self, epochs=50):
        """Run architecture search"""
        print("Starting DARTS architecture search...")
        
        for epoch in range(epochs):
            train_iter = iter(self.train_loader)
            val_iter = iter(self.val_loader)
            
            epoch_w_loss = 0
            epoch_a_loss = 0
            num_steps = 0
            
            while True:
                try:
                    train_data = next(train_iter)
                    val_data = next(val_iter)
                    
                    w_loss, a_loss = self.train_step(train_data, val_data)
                    
                    epoch_w_loss += w_loss
                    epoch_a_loss += a_loss
                    num_steps += 1
                    
                except StopIteration:
                    break
            
            avg_w_loss = epoch_w_loss / num_steps
            avg_a_loss = epoch_a_loss / num_steps
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Weight Loss: {avg_w_loss:.4f}, "
                  f"Arch Loss: {avg_a_loss:.4f}")
        
        # Extract final architecture
        final_arch = self.model.get_architecture()
        print("\nDiscovered Architecture:")
        for layer_info in final_arch:
            print(f"  {layer_info}")
        
        return final_arch

# Example usage
if __name__ == "__main__":
    # Create DARTS network
    model = DARTSNetwork(C=16, num_classes=10, layers=8)
    
    print("DARTS Network initialized with searchable architecture")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
