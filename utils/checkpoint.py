import torch
import os


def save_checkpoint(config, epoch, model, optimizer, best_acc1, scheduler=None):
    save_states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc1': best_acc1,
    }

    if scheduler is not None:
        save_states['scheduler'] = scheduler.state_dict()

    torch.save(save_states, os.path.join(config.result_path, 'checkpoint.pth'))


def resume(config, model, optimizer, scheduler=None):

    resume_path = os.path.join(config.result_path, 'checkpoint.pth')
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(
        resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])

    # confirm whether the optimizer matches that of checkpoints
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, model, optimizer, best_acc1, scheduler
