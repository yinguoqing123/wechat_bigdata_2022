import logging
import os
from pickletools import optimize
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from wx_uni_model import WXUniModel
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from create_optimizer import create_optimizer, get_reducelr_schedule, get_warmup_schedule

def validate(model, val_dataloader):
    model.eval()
    losses = []
    accuracy_mlm = []
    accuracy_itm = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, accuracy_mlm, accuracy_itm = model(batch, pretrain=True)
            loss = loss.mean()
            accuracy_mlm.append(accuracy_mlm.cpu().numpy())
            accuracy_itm.append(accuracy_itm.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    accuracy_mlm = sum(accuracy_mlm) / len(accuracy_mlm)
    accuracy_itm = sum(accuracy_itm) / len(accuracy_itm)
    model.train()
    return loss, accuracy_mlm, accuracy_itm


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args, pretrain=True)

    # 2. build model and optimizers
    model = WXUniModel(task=['mlm', 'itm'], model_path=args.bert_dir)
    # optimizer, scheduler = build_optimizer(args, model)
    optimizer = create_optimizer(model)
    scheduler_warmup = get_warmup_schedule(optimizer, num_warmup_steps=args.bert_warmup_steps)
    scheduler_reducelr = get_reducelr_schedule(optimizer, mode='min')
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        # model = model.to(args.device)

    # 3. training
    step = 0
    best_score = args.best_score
    min_loss = float('inf')
    start_time = time.time()
    accumulation_steps = 4
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            step += 1
            model.train()
            loss, accuracy_mlm, accuracy_itm = model(batch, pretrain=True)
            loss = loss.mean() / accumulation_steps
            accuracy = accuracy.mean()
            loss.backward() 
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler_warmup.step()
            
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy_mlm {accuracy_mlm:.3f}, accuracy_itm {accuracy_itm:.3f}")
            if step % 2000 == 0:
                # 4. validation
                loss, accuracy_mlm, accuracy_itm = validate(model, val_dataloader)
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, accuracy_mlm {accuracy_mlm:.3f}, accuracy_itm: {accuracy_itm:.3f}")

                # 5. save checkpoint
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss, 'accuracy_mlm': accuracy_mlm, 'accuracy_itm': accuracy_itm},
                            f'{args.savedmodel_path}/model_best_pretrain.bin')
                
                if step > args.bert_warmup_steps:
                    scheduler_reducelr.step(loss.item())
                    


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
