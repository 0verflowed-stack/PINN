# PINN (Physics-Informed Neural Network)
Code for solving various PDEs with PINN 

Here is a list of equations that it solves:
1. File 1DLinearPINN.py solves Puasson's equation:
<!-- $$
-(k(x)'p)'= f(x), x \in (0,1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/AlkEIpFZSv.svg"></div> 
<!-- $$
p(0)=p(1)=0.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/dfNzBmzUPr.svg"></div>
2. File 1DNonLinearPINN.py solves Puasson's equation:
<!-- $$
-(k(x,p)'p)'= f(x), x \in (0,1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/UXFvBjygeM.svg"></div>
<!-- $$
p(0)=p(1)=0.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/SzldDtuys0.svg"></div>
3. File 2DPlus1DPlot.py solves Puasson's equation:
<!-- $$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/WcLZx79BhP.svg"></div>
<!-- $$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_3,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/FCj1f3nGZP.svg"></div>
<!-- $$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_2\text{ i }\Gamma_4.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/5BgANAmuPI.svg"></div>
4. File 2DPlus1DPlotZeroF.py solves Laplace's equation:
<!-- $$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/p5GYj3MBXM.svg"></div>
<!-- $$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_3,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/IOyklTaMeZ.svg"></div>
<!-- $$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_2\text{ i }\Gamma_4.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/FFFSWJTJVH.svg"></div>
5. File 2DPlus1DPlotZeroFHalfDirichlet.py solves Laplace's equation:
<!-- $$
\bigtriangleup u = f(x), \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/BvIhrzxLsv.svg"></div>
<!-- $$
u = 0\text{ на }\Gamma_1\text{ i }\Gamma_4,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/acTqFsBb6D.svg"></div>
<!-- $$
u = 1\text{ на }\Gamma_2\text{ i }\Gamma_5,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/f6kAFBA4Yc.svg"></div>
<!-- $$
\frac{\partial u}{\partial n} = 0\text{ на }\Gamma_3\text{ i }\Gamma_6.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/PTMxNKJzdt.svg"></div>
<p align="center">
    <img src="./imgs/ZeroFHalfDirichletRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

