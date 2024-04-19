# PINN (Physics-Informed Neural Network)
Code for solving various PDEs with PINN.

Solutions are compared to exact solution or FEM solution.

Here is a list of equations that are solved:
1. File 1DLinearPINN.py solves 1D linear Puasson's equation:
<!-- $$
-(k(x)'p)'= f(x), x \in (0,1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/uMGXBfDa3x.svg"></div>

<!-- $$
p(0)=p(1)=0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/ELxGJC4mqm.svg"></div> 


<!-- $$
k(x,p)=1+xp,f(x)=-1.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/Bj2TuJXC7V.svg"></div>


2. File 1DNonLinearPINN.py solves 1D Nonlinear Puasson's equation:
<!-- $$
-(k(x,p)'p)'= f(x), x \in (0,1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/5hGc6K10kU.svg"></div>

<!-- $$
p(0)=p(1)=0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/PKw2Nh043B.svg"></div>
<!-- $$
k(x)=\frac{1}{4+sin(2\pi x)}, f(x)=1.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/H952OpsxZA.svg"></div>

3. File 2DPlus1DPlot.py solves 2D linear Puasson's equation:
<!-- $$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/SOZLn6RvvI.svg"></div>

<!-- $$
u = 0\text{ on }\Gamma_1\text{ and }\Gamma_3,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/MZNytBJivg.svg"></div>


<!-- $$
\frac{\partial u}{\partial n} = 0\text{ on }\Gamma_2\text{ and }\Gamma_4,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/gDKJpOmmqy.svg"></div>
<!-- $$
f(x)=-\pi sin(\pi x_1)sin(\pi x_2).
$$ --> 

<div align="center"><img style="background: white;" src="./svg/b3etJtqMtT.svg"></div>

<p align="center">
    <img src="./imgs/SquareRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

and analogous 1D linear Puasson's equation:

<!-- $$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/iTGZcVDG5m.svg"></div>

<!-- $$
u(0) = u(1) = 0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/OzBpoNSjef.svg"></div>
<!-- $$
f(x)=-\pi sin(\pi x).
$$ --> 

<div align="center"><img style="background: white;" src="./svg/s2TT44Q9Bl.svg"></div>

4. File 2DPlus1DPlotZeroF.py solves 2D linear Laplace's equation:
<!-- $$
\bigtriangleup u = 0, \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/W5jB3aIZ3W.svg"></div>

<!-- $$
u = 1\text{ on }\Gamma_1, u = 0\text{ on }\Gamma_3,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/c6PnjswTHo.svg"></div>

<!-- $$
\frac{\partial u}{\partial n} = 0\text{ on }\Gamma_2\text{ and }\Gamma_4.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/9d39L8EiBa.svg"></div>

<p align="center">
    <img src="./imgs/SquareRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

and analogous 1D linear Laplace's equation:
<!-- $$
\bigtriangleup u = 0, \quad x \in \Omega = (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/wP04iSRVdp.svg"></div>

<!-- $$
u = 1\text{ when } x = 0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/xRb21CObXQ.svg"></div>
<!-- $$
u = 0\text{ when } x = 1.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/vvo74wBUZf.svg"></div>

5. File 2DPlus1DPlotZeroFHalfDirichlet.py solves 2D linear Laplace's equation:
<!-- $$
\bigtriangleup u = 0, \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/sQulDMmPK3.svg"></div>

<!-- $$
u = 0\text{ on }\Gamma_1\text{ and }\Gamma_4,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/yLZ0U7Yw5v.svg"></div>

<!-- $$
u = 1\text{ on }\Gamma_2\text{ and }\Gamma_5,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/iMNjgPPF0C.svg"></div>

<!-- $$
\frac{\partial u}{\partial n} = 0\text{ on }\Gamma_3\text{ and }\Gamma_6.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/bGLke2iuxl.svg"></div>

<p align="center">
    <img src="./imgs/ZeroFHalfDirichletRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

6. File 2DRobin.ipynb solves Puasson's/Laplace's 2D equation:

$$
\bigtriangleup u = 0, \quad x \in \Omega = (0, 1) \times (0, 1),
$$

$$
u = 0\text{ on }\Gamma_1, 
$$
$$
\frac{\partial u}{\partial n} = \alpha u\text{ on }\Gamma_2, 
$$
$$
u = 1\text{ on }\Gamma_3, 
$$
$$
\frac{\partial u}{\partial n} = 0\text{ on }\Gamma_4.
$$

<p align="center">
    <img src="./imgs/SquareRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

7. File 1DNonStationaryHeatEquation.ipynb solves non-stationary 1D heat equation:

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}, \quad (x,t) \in \Omega = (0, 1) \times (0, 1),
$$

$$
u(0, t) = 0,
$$

$$
u(1, t) = 0,
$$

$$
u(x, 0) = \sin(\pi x) - \sin(2 \pi x) + \sin(3 \pi x).
$$

8. File 2DNonStationaryHeatEquation.ipynb solves non-stationary 2D heat equation:

$$
\frac{\partial u}{\partial t} = \bigtriangleup u, \quad (x,t) \in \Omega = (0, 1)^2\times (0, 1),
$$

$$
u(0, x_2, t) = u(x_{1 max},x_2, t) = u(x_1, 0, t) = u(x_1, x_2, 0) = 0,
$$

$$
u(x_1, x_{2 max}) = 1.
$$

<!-- $$
u((x_1, x_2), 0) = \sin(\pi x_1) - \sin(2 \pi x_2) + \sin(3 \pi x_1).
$$ -->