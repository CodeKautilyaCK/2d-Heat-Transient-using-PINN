# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Set device --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Neural Network --------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# -------------------- PDE Residual --------------------
def pde_residual(model, x, y, t, alpha):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    T = model(torch.cat([x, y, t], dim=1))
    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    residual = T_t - alpha * (T_xx + T_yy)
    return residual

# -------------------- Training Data --------------------
def generate_training_points(nx=20, ny=20, nt=20, Lx=1.0, Ly=1.0, T_end=1.0):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    t = np.linspace(0, T_end, nt)
    X, Y, T = np.meshgrid(x, y, t)
    X_flat = X.flatten()[:, None]
    Y_flat = Y.flatten()[:, None]
    T_flat = T.flatten()[:, None]

    return torch.tensor(X_flat, dtype=torch.float32).to(device), \
           torch.tensor(Y_flat, dtype=torch.float32).to(device), \
           torch.tensor(T_flat, dtype=torch.float32).to(device)

# -------------------- Initial and Boundary Conditions --------------------
def initial_condition(x, y):
    # Example: hotspot in center
    return torch.exp(-50*((x-0.5)**2 + (y-0.5)**2))

def boundary_condition(x, y, t):
    # Example: Dirichlet 0 at edges
    bc = torch.zeros_like(x)
    mask = (x==0) | (x==1) | (y==0) | (y==1)
    bc[mask] = 0.0
    return bc

# -------------------- Training --------------------
def train(model, optimizer, epochs, alpha):
    x, y, t = generate_training_points()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE Residual
        res = pde_residual(model, x, y, t, alpha)
        loss_pde = torch.mean(res**2)
        
        # IC Loss
        x0 = torch.linspace(0,1,20)[:,None].to(device)
        y0 = torch.linspace(0,1,20)[:,None].to(device)
        t0 = torch.zeros_like(x0).to(device)
        T0 = initial_condition(x0, y0)
        T_pred0 = model(torch.cat([x0, y0, t0], dim=1))
        loss_ic = torch.mean((T_pred0 - T0)**2)

        # BC Loss
        x_bc = torch.linspace(0,1,20)[:,None].to(device)
        y_bc = torch.linspace(0,1,20)[:,None].to(device)
        t_bc = torch.linspace(0,1,20)[:,None].to(device)
        T_bc = boundary_condition(x_bc, y_bc, t_bc)
        T_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
        loss_bc = torch.mean((T_pred_bc - T_bc)**2)

        # Total loss
        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -------------------- Main --------------------
if __name__ == "__main__":
    layers = [3, 50, 50, 50, 50, 1]
    model = PINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    alpha = 0.01
    epochs = 2000

    train(model, optimizer, epochs, alpha)

    # Predict temperature at t = 0.5
    nx, ny = 50, 50
    x = torch.linspace(0,1,nx).to(device)[:,None]
    y = torch.linspace(0,1,ny).to(device)[:,None]
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze())
    t_pred = 0.5 * torch.ones_like(X).to(device)
    T_pred = model(torch.cat([X.reshape(-1,1), Y.reshape(-1,1), t_pred.reshape(-1,1)], dim=1))
    T_pred = T_pred.detach().cpu().numpy().reshape(nx, ny)

    plt.imshow(T_pred, extent=[0,1,0,1], origin='lower', cmap='hot')
    plt.colorbar()
    plt.title("Temperature at t=0.5")
    plt.show()
