# Copyright 2018 Rajat Modi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import math

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

#
''' LS Smiths cyclic learning rate policy was ok
Another concept is annealing where the lr varies through half a cosine_warmup
The notion is to find a trough region in the loss graph where generalization num_workers
If we are in steep minima, increase lr is beneficial as the model might go to a flatter region which is more stable
That is the notion behind SGD with warm restarts.
'''
class HalfCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.T_max = T_max
        super(HalfCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % (2 * self.T_max) < self.T_max:
            cos_unit = 0.5 * \
                (math.cos(math.pi * self.last_epoch / self.T_max) - 1)
        else:
            cos_unit = 0.5 * \
                (math.cos(math.pi * (self.last_epoch / self.T_max - 1)) - 1)

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * 1.0e-4
            range = math.log10(base_lr - math.log10(min_lr))
            lrs.append(10 ** (math.log10(base_lr) + range * cos_unit))
        return lrs


def get_scheduler(optimizer, config):
    if config.scheduler.name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    elif config.scheduler.name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, config.scheduler.params.t_max, eta_min=1e-6,
                                      last_epoch=-1)
    elif config.scheduler.name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.params.t_max, eta_min=1e-6,
                                                last_epoch=-1)
    elif config.scheduler.name == 'half_cosine':
        scheduler = HalfCosineAnnealingLR(
            optimizer, config.scheduler.params.t_max, last_epoch=-1)
    else:
        scheduler = None
    return scheduler
