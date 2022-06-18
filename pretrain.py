import logging
import os
from pickletools import optimize
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from pretrain_model import WXUniPretrainModel
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from create_optimizer import create_optimizer, get_reducelr_schedule, get_warmup_schedule
from util import FGM, EMA
from transformers import get_cosine_schedule_with_warmup
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

def validate(model, val_dataloader):
    model.eval()
    mlm_losses, itm_losses = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            mlm_loss, itm_loss, _, _, vm_loss = model(batch)
            mlm_losses.append(mlm_loss.mean().to('cpu').item())
            itm_losses.append(itm_loss.mean().to('cpu').itm())
            
    model.train()
    return sum(mlm_losses)/len(mlm_losses) + sum(itm_losses)/len(itm_losses)


def pretrain(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args, pretrain=False)
    
    # 2. build model and optimizers
    model = WXUniPretrainModel(args, use_arcface_loss=False)
    # optimizer, scheduler = build_optimizer(args, model)
    optimizer = create_optimizer(model)
    scheduler_warmup = get_warmup_schedule(optimizer, num_warmup_steps=args.warmup_steps)
    scheduler_reducelr = get_reducelr_schedule(optimizer, patience=1, factor=0.6)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=20000 * 5)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        # model = model.to(args.device)
        
    # 3. training
    step = 0
    loss_min = float('inf')
    start_time = time.time()
    accumulation_steps = 1
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            step += 1
            model.train()
            mlm_loss, itm_loss, mlm_accuracy, itm_accuracy, vm_loss = model(batch)
            mlm_loss = mlm_loss.mean() / accumulation_steps
            itm_loss = itm_loss.mean() / accumulation_steps
            vm_loss = vm_loss.mean() / accumulation_steps
            mlm_accuracy = mlm_accuracy.mean()
            itm_accuracy = itm_accuracy.mean()
            loss = mlm_loss + itm_loss + vm_loss
            loss.backward() 
                
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if step < args.warmup_steps:
                    scheduler_warmup.step()
                # scheduler.step()
                
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: mlm_loss {mlm_loss:.3f} itm_loss {itm_loss:.3f} mlm accuracy {mlm_accuracy:.3f} itm accuracy {itm_accuracy:.3f}")
                
            # if step % 1000 == 0:
            #     # 4. validation  
            #     loss, results = validate(model, val_dataloader)
            #     results = {k: round(v, 4) for k, v in results.items()}
            #     logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            
            if step % 3000 == 0:
                # 5. save checkpoint
                val_loss = validate(model, val_dataloader)
                if step > args.warmup_steps:
                    scheduler_reducelr.step(val_loss)
                    
                if val_loss < loss_min:
                    loss_min = val_loss
                    logging.info(f"val loss: {val_loss:.3f} step {step} lr {optimizer.param_groups[0]['lr']} {optimizer.param_groups[-1]['lr']}")
                    print(f"正在保存模型, val loss: {val_loss}")
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'step': step, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss': itm_loss},
                            f'{args.savedmodel_path}/model_best.bin')
            if step % 10000 == 0:  
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss': itm_loss},
                                    f'{args.savedmodel_path}/model_best_step{step}.bin')

                    
 

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    pretrain(args)


if __name__ == '__main__':
    main()
