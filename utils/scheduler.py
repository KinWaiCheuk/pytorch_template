import math
from torch.optim.lr_scheduler import _LRScheduler

class TriStageLRSchedule(_LRScheduler):
    """
    Scheldule modified from fairseq
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py#L51
    """

    def __init__(self, optimizer, lr_phases, phase_ratio, max_update, last_epoch=-1, final_lr_scale=0.05, verbose=False):
        # calculate LR at each point
        self.peak_lr = lr_phases[1]
        self.init_lr = lr_phases[0]
        self.final_lr = lr_phases[2]
        self.optimizer = optimizer
        self._step_count = 1

        if phase_ratio is not None:
            assert max_update > 0
            assert sum(phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(max_update * phase_ratio[0])
            self.hold_steps = int(max_update * phase_ratio[1])
            self.decay_steps = int(max_update * phase_ratio[2])
        else:
            raise ValueError("Please specify phase_ratio")

        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
#         self.decay_factor = -final_lr_scale / self.decay_steps

        # initial learning rate
        self.lr = self.init_lr
        self.optimizer.param_groups[0]['lr']=self.lr
#         self.optimizer.set_lr(self.lr)
        super().__init__(optimizer, last_epoch, verbose)

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def step(self):
        """Update the learning rate at the end of the given epoch."""
        self.step_update(self._step_count)
        # we don't change the learning rate at epoch boundaries
        self._step_count += 1
        

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(num_updates)
        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
#             self.lr = self.peak_lr - self.decay_factor * steps_in_stage
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.optimizer.param_groups[0]['lr']=self.lr
#         self.optimizer.set_lr(self.lr)
        
        return self.lr