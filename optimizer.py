import torch, math


class PrivateDiff(torch.optim.Optimizer):

    def __init__(self, params, loss_fn, lr, lr_alpha, c1, c2, c_y, sigma1, sigma2, sigma_alpha, inner_iters, T) -> None:
        try:
            self.a = loss_fn.a
            self.b = loss_fn.b
            self.alpha = loss_fn.alpha
        except:
            print("AUCMLoss is not found!")

        self.params = list(params)
        self.lr = lr
        self.lr_alpha = lr_alpha
        self.c1 = c1
        self.c2 = c2
        self.c_y = c_y
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma_alpha = sigma_alpha
        self.inner_iters = inner_iters
        self.T = T
        if self.a is not None and self.b is not None:
            self.params = self.params + [self.a, self.b]

        defaults = dict(
            lr=self.lr,
            a=self.a,
            b=self.b,
            alpha=self.alpha,
        )
        super().__init__(self.params, defaults)

    def __setstate__(self, state):
        super(PrivateDiff, self).__setstate__(state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.alpha.grad = None
        return super().zero_grad(set_to_none)

    @torch.no_grad()
    def clip_gradient(self, grad, clip_value):
        norm = torch.linalg.norm(grad)
        if norm == 0:
            return grad
        coef = clip_value / norm
        coef = torch.minimum(coef, torch.ones_like(coef, device=coef.device))
        return coef * grad

    @torch.no_grad()
    def dpgdsc(self, sigma, lr, iters, closure):
        for _ in range(iters):
            closure()
            if self.alpha.grad is None:
                continue
            grad = self.alpha.grad.data
            grad = self.clip_gradient(grad, self.c_y)
            self.alpha.data.add_(grad, alpha=lr)
            self.zero_grad()
        self.alpha.data.add_(torch.normal(0, math.sqrt(sigma), self.alpha.shape, device=self.alpha.device))

    @torch.no_grad()
    def step(self, r, closure=None):
        assert closure is not None, "DIFF2MINMAX requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        # NOTE: Line 2
        self.dpgdsc(self.sigma_alpha, self.lr_alpha, self.inner_iters, closure)

        closure()
        # NOTE: Line 3
        if r % self.T == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # NOTE: Line 4
                    grad = p.grad.data
                    param_state = self.state[p]
                    param_state["gradient_buffer"] = torch.clone(grad).detach()

                    c1 = self.c1
                    d_r = self.clip_gradient(grad, c1)

                    # NOTE: Line 4
                    v = 0

                    # NOTE: Line 6
                    v = d_r + v + torch.normal(0, math.sqrt(self.sigma1 * c1), p.shape, device=grad.device)

                    # NOTE: Line 7
                    param_state["param_buffer"] = torch.clone(p).detach()
                    p.data = p.data.add_(v, alpha=-self.lr)

                    # NOTE: Save the gradient and params and the update
                    param_state["update_buffer"] = torch.clone(v).detach()

        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # NOTE: Line 5
                    grad = p.grad.data
                    param_state = self.state[p]

                    c2 = self.c2 * torch.linalg.norm(p.data - param_state["param_buffer"])
                    d_r = self.clip_gradient(grad - param_state["gradient_buffer"], c2)

                    # NOTE: Line 6
                    v = param_state["update_buffer"]
                    v = d_r + v + torch.normal(0, math.sqrt(self.sigma2 * c2), p.shape, device=grad.device)

                    # NOTE: Line 7
                    param_state["param_buffer"] = torch.clone(p).detach()
                    p.data = p.data.add_(v, alpha=-self.lr)

                    # NOTE: Save the gradient and params and the update
                    param_state["gradient_buffer"] = torch.clone(grad).detach()
                    param_state["update_buffer"] = torch.clone(v).detach()

        self.zero_grad()

