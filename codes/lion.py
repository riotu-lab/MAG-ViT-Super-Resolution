import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Implementation of the Lion optimizer."""

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight decay: {weight_decay}"
            )

        defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for parameter in group['params']:
                if parameter.grad is None:
                    continue

                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        'Lion does not support sparse gradients'
                    )

                gradient = parameter.grad
                state = self.state[parameter]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(parameter)

                exp_avg = state['exp_avg']

                # Decoupled weight decay
                parameter.mul_(
                    1.0 - group['lr'] * group['weight_decay']
                )

                # Parameter update
                update = exp_avg.mul(beta1).add(
                    gradient,
                    alpha=1.0 - beta1
                )

                parameter.add_(
                    update.sign(),
                    alpha=-group['lr']
                )

                # Momentum update
                exp_avg.mul_(beta2).add_(
                    gradient,
                    alpha=1.0 - beta2
                )

        return loss
