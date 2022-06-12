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
from util import FGM, EMA

def validate(model, val_dataloader):
    model.eval()
    predictions_text, predictions_frame, predictions_union = [], [], []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, text_result, frame_result, union_result = model(batch)
            loss = loss.mean()
            predictions_text.extend(text_result['pred_label_id'].cpu().numpy())
            predictions_frame.extend(frame_result['pred_label_id'].cpu().numpy())
            predictions_union.extend(union_result['pred_label_id'].cpu().numpy())
            labels.extend(text_result['label_lv2'].cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results_text = evaluate(predictions_text, labels, name='text')
    results_frame = evaluate(predictions_frame, labels, name='frame')
    results_union = evaluate(predictions_union, labels, name='union')
    results_text.update(results_frame)
    results_text.update(results_union)
    
    model.train()
    return loss, results_text


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    
    # 2. build model and optimizers
    model = WXUniModel(args, task=[], use_arcface_loss=False)
    fgm = FGM(model)
    ema = EMA(model)
    first_ema_flag = True
    # optimizer, scheduler = build_optimizer(args, model)
    optimizer = create_optimizer(model)
    scheduler_warmup = get_warmup_schedule(optimizer, num_warmup_steps=args.bert_warmup_steps)
    scheduler_reducelr = get_reducelr_schedule(optimizer, patience=1)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        # model = model.to(args.device)
        
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    accumulation_steps = 4
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            step += 1
            model.train()
            loss, _, _, union_result = model(batch)
            loss = loss.mean() / accumulation_steps
            accuracy = union_result['accuracy'].mean()
            loss.backward() 
            # if step > 3000:
            #     fgm.attack()
            #     loss_adv, _, _, union_result = model(batch)
            #     accuracy = union_result['accuracy'].mean()
            #     loss_adv.backward()
            #     fgm.restore()
                
            if step > 7000 and first_ema_flag:
                ema.register()
                first_ema_flag = False
            
                
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler_warmup.step()
                if step > 7000:
                    ema.update()
                
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
                
            if step % 1000 == 0:
                # 4. validation
                if step > 7000:
                    ema.apply_shadow()
                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                
                # 5. save checkpoint
                mean_f1 = results['mean_f1_union']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                            f'{args.savedmodel_path}/model_best.bin')
                
                if step > 7000:
                    ema.restore()
                    
                if step > args.bert_warmup_steps:
                    scheduler_reducelr.step(mean_f1)
                    


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
