# Stein Density Ratio Estimation (SDRE) and Its Applications

### Reference: 
[*Song Liu, Takafumi Kanamori, Wittawat Jitkrittum, Yu Chen, Fisher Efficient Inference of Intractable Models, E-print: arXiv:1805.07454, 2019*](https://arxiv.org/abs/1805.07454)

### Install the `sdre` package

If you plan to modify our code (very likely, you will want to do so), it is best to install by:

1. Clone this repository 
2. `cd` to the folder that you get, and install our package by (notice the dot at the end)

        pip install -e .

There an alternative way to install without cloning. But we do not recommend at
this point since the code requires direct modification at this point.


Once installed, you should be able to do `import sdre` in a Python shell without any error.

### Folder Structure: 
- **sdre**: the provided Python package. 
- **script/DRE**: Stein Density Ratio Estimation
- **script/Inference**: Model Inference using SDRE

- README: this file

To run primal Stein density ratio estimation:

```bash
python script/DRE/demo.py
```

```
0
1
2
0 delta: [-0.00056058  0.00018254 -0.00041193]
100 delta: [-0.05309547  0.01852391 -0.03606012]
200 delta: [-0.09931757  0.03469031 -0.06197491]
...

```

![demo.png](demo.png "")

