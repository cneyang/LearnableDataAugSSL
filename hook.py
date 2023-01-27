import copy
import torch
import torch.nn.functional as F
import numpy as np

from semilearn.core.hooks import Hook

class PolicyUpdateHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def policy_update(self, algorithm, loss):
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            algorithm.loss_scaler.step(algorithm.policy_optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            algorithm.policy_optimizer.step()

        # algorithm.policy_scheduler.step()
        algorithm.policy_optimizer.zero_grad()
    
    def policy_step(self, algorithm):
        if algorithm.use_amp:
            algorithm.loss_scaler.step(algorithm.policy_optimizer)
            algorithm.loss_scaler.update()
        else:
            algorithm.policy_optimizer.step()

        algorithm.policy_optimizer.zero_grad()

class AdversarialAttackHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, algorithm, x_ulb_w, targets_ulb_w, y_ulb_w, feats):
        mu = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mu = torch.tensor(mu).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        upper_limit = ((1 - mu) / std).cuda(algorithm.gpu)
        lower_limit = ((0 - mu) / std).cuda(algorithm.gpu)

        perturbations = torch.zeros_like(x_ulb_w)
        perturbations.uniform_(-0.01, 0.01)
        perturbations.data = self.clamp(perturbations, lower_limit - x_ulb_w, upper_limit - x_ulb_w)
        perturbations.requires_grad = True

        lams = np.logspace(0, 1, algorithm.attack_iters)
        for it in range(algorithm.attack_iters):
            step_size = algorithm.bound * (1/lams[it])
            lam = algorithm.attack_lam * lams[it]

            if perturbations.grad is not None:
                perturbations.grad.data.zero_()

            x_adv = x_ulb_w + perturbations

            outs_adv = algorithm.model(x=x_adv, adv=True)
            logits_adv, feats_adv = outs_adv['logits'], outs_adv['feats']
            prob_adv = torch.softmax(logits_adv / algorithm.T, dim=-1)
            y_adv = torch.log(torch.gather(prob_adv, 1, targets_ulb_w.view(-1, 1)).squeeze(dim=1))

            pip = (algorithm.normalize_flatten_features(feats_adv) - feats).norm(dim=1).mean()
            constraint = y_ulb_w - y_adv
            loss = -pip + lam * F.relu(constraint - algorithm.bound).mean()
            loss.backward()
            algorithm.optimizer.zero_grad()

            grad = perturbations.grad.data
            grad_normed = grad / (grad.reshape(grad.size()[0], -1).norm(dim=1)[:, None, None, None] + 1e-8)

            with torch.no_grad():
                y_after = torch.log(torch.gather(torch.softmax(algorithm.model(x=x_ulb_w + perturbations - grad_normed * 0.1, adv=True)['logits'] / algorithm.T, dim=1), 1, targets_ulb_w.view(-1, 1)).squeeze(dim=1))
                dist_grads = torch.abs(y_adv - y_after) / 0.1
                norm = step_size / (dist_grads + 1e-4)
            perturbation_updates = - grad_normed * norm[:, None, None, None]
            perturbations.data = self.clamp(perturbations + perturbation_updates, lower_limit - x_ulb_w, upper_limit - x_ulb_w).detach()

        return (x_ulb_w + perturbations).detach()

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

class TimeConsistencyHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def memory_update(self, algorithm):
        if algorithm.mem_update.all():
            print('Updated at {}th iteration'.format(algorithm.it))
            _, indices = torch.sort(algorithm.mem_tc, descending=True)
            kt = (1 - algorithm.tau) * algorithm.args.ulb_dest_len
            mem_tc_copy = copy.deepcopy(algorithm.mem_tc)
            algorithm.threshold = mem_tc_copy[indices[int(kt)]]
            algorithm.mem_update = torch.zeros_like(algorithm.mem_update, dtype=torch.bool)
