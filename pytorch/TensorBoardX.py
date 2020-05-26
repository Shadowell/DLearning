from tensorboardX import SummaryWriter

writer = SummaryWriter('logs/tmp')
writer.add_scalar('logss/total_loss', loss.data[0], total_iter)
writer.add_scalar('loss/rpn_loss', rpn_loss.data[0], total_iter)