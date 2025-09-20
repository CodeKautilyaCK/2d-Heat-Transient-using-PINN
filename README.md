# 2d Heat Tranisent using PINN's #

Problem:
Equation: ∂T∂t=α(∂2T∂x2+∂2T∂y2)
Domain: 1x1 plate, hotspot at center
BCs: Fixed edges, insulated edges
Goal: Predict T(x,y,t)T(x,y,t)T(x,y,t) continuously

Training:
2000 epochs, loss decreased from 0.192 → 0.0007 ✅
Collocation points: 8000
Optimizer: Adam

Validation:
Compared with classical finite difference solution on same grid
RMS error: 0.0032
Max absolute error: 0.007
Heatmaps overlay showed near-perfect agreement at t = 0.1, 0.5, 1.0

Key Takeaways:
- PINNs can learn PDE solutions continuously without remeshing
- Validation with traditional methods ensures accuracy and reliability
