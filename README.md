# PINN (Physics-Informed Neural Network)
Code for solving various PDEs with PINN 

Here is a list of equations that it solves:
1. File 1DLinearPINN.py solves Puasson's equation:
$$
-(k(x)'p)'= f(x), x \in (0,1),
$$
$$
p(0)=p(1)=0.
$$
2. File 1DNonLinearPINN.py solves Puasson's equation:
$$
-(k(x,p)'p)'= f(x), x \in (0,1),
$$
$$
p(0)=p(1)=0.
$$
3. File 2DPlus1DPlot.py solves Puasson's equation:
$$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$
$$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_3,
$$
$$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_2\text{ i }\Gamma_4.
$$
4. File 2DPlus1DPlotZeroF.py solves Laplace's equation:
$$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$
$$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_3,
$$
$$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_2\text{ i }\Gamma_4.
$$
5. File 2DPlus1DPlotZeroFHalfDirichlet.py solves Laplace's equation:
$$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$
$$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_4,
$$
$$
u = 1\text{ на }\Gamma_2\text{ i }\Gamma_5,
$$
$$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_3\text{ i }\Gamma_6.
$$
<p align="center">
<img src="./imgs/ZeroFHalfDirichletRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

