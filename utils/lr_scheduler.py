import torch.optim.lr_scheduler as lr_scheduler

class LRScheduler(object):

    def __init__(self, num_epochs, scheduler_type ='linear', start_schedule_epoch=0):
        self.lr_schedulers = []
        self.scheduler_type = scheduler_type
        self.start_schedule_epoch = start_schedule_epoch
        self.num_epochs = num_epochs

    def reg_optimizer(self, optimizer):
        if self.scheduler_type  == 'linear':
            def lambda_rule(epoch):
                start_schedule_epoch = self.start_schedule_epoch - 1
                return 1.0 - (max(0, epoch-start_schedule_epoch) / (self.num_epochs-start_schedule_epoch+1))

            s = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            self.lr_schedulers.append(s)

    def step(self):
        for s in self.lr_schedulers:
            s.step()