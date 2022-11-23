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

        algorithm.policy_scheduler.step()
        algorithm.policy_optimizer.zero_grad()
