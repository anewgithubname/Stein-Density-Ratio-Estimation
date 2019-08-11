# Stein Density Ratio Estimation (SDRE) and Its Applications

### Reference: 
[*Song Liu, Takafumi Kanamori, Wittawat Jitkrittum, Yu Chen, Fisher Efficient Inference of Intractable Models, E-print: arXiv:1805.07454, 2019*](https://arxiv.org/abs/1805.07454)

### Folder Structure: 
- **DRE**: Stein Density Ratio Estimation
- **Inference**: Model Inference using SDRE
- **util**: Helper Functions

- README: this file

To run primal Stein density ratio estimation:

```bash
python DRE/demoPrimal.py
```

```
0
1
2
3
4
delta: [-0.01534369 -0.01241881 -0.01777918 -0.00768591 -0.0086371 ]
delta: [-0.02901659 -0.02353518 -0.03350215 -0.01435563 -0.01642222]
delta: [-0.04125819 -0.03353829 -0.04745818 -0.02016631 -0.02346772]
...
gradient descent converged after 90 iterations

 diff between GD solver and builtin solver:
[-1.74338949e-06  2.20755595e-06 -3.80123256e-06  5.16225588e-06
 -5.94081083e-06]
```