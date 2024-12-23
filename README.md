# Improved Rates of Differentially Private Nonconvex-Strongly-Concave Minimax
[Ruijia Zhang](https://richard-zzz.github.io/)&ast;, [Mingxi Lei](https://mingxilei.github.io)&ast;, [Meng Ding](https://meng-ding.github.io), [Zihang Xiang](https://zihangxiang.github.io), [Jinhui Xu](https://cse.buffalo.edu/~jinhui/), [Di Wang](https://shao3wangdi.github.io), "Improved Rates of Differentially Private Nonconvex-Strongly-Concave Minimax", AAAI, 2025.

> **Abstract:** In this paper, we study the problem of (finite sum) minimax optimization in the Differential Privacy (DP) model. Unlike most of the previous studies on the (strongly) convex-concave settings or loss functions satisfying the Polyak-\L{ojasiewicz} condition, here we mainly focus on the nonconvex-strongly-concave one, which encapsulates many models in deep learning such as deep AUC maximization. Specifically, we first analyze a DP version of Stochastic Gradient Descent Ascent (SGDA) and show that it is possible to get an $(\epsilon,\delta)$-DP estimator whose $l_2$-norm of the gradient for the empirical risk function is upper bounded by $\tilde{O}(\frac{d^{1/4}}{({n\epsilon})^{1/2}})$, where $d$ is the model dimension and $n$ is the sample size. We then propose a new method with less gradient noise variance and improve the upper bound to $\tilde{O}(\frac{d^{1/3}}{(n\epsilon)^{2/3}})$, which matches the best-known result for DP Empirical Risk Minimization with non-convex loss. We also discussed several lower bounds of private minimax optimization. Finally, experiments on AUC maximization, generative adversarial networks, and temporal difference learning with real-world data support our theoretical analysis. 

## Dependencies

- Python
- PyTorch 
- TorchVision
- LibAUC
- NumPy
- PIL

## PrivateDiff-Minimax Algorithm
![RDN](/figure/algorithm.png)
## Usage

```python
from optimizer import PrivateDiff
...

model = YourModel()
optimizer = PrivateDiff(model.parameters(), ...)
...

r=0
for input, output in data:
  def closure():
    loss = loss_function(output, model(input))
    loss.backward()
    return loss

  loss = loss_function(output, model(input))
  loss.backward()
  optimizer.step(r=r, closure)
  optimizer.zero_grad()
  r += 1
...
```

<br>

## Reproducibility

To reproduce the AUC performance on the Imbalanced MNIST dataset (with the privacy budget of $\epsilon=0.5$) as reported in Table 1, run main.py.

```
python main.py
```

![RDN](/figure/performance.png)

## Citation

TBD.
