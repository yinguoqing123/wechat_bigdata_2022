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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

def validate(model, val_dataloader):
    model.eval()
    predictions_text, predictions_frame, predictions_union, predictions_mix = [], [], [], []
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


def pretrain(args):
    # 1. load data
    train_dataloader = create_dataloaders(args)
    
    # 2. build model and optimizers
    model = WXUniModel(args, task=[], use_arcface_loss=True)
    # optimizer, scheduler = build_optimizer(args, model)
    optimizer = create_optimizer(model)
    scheduler_warmup = get_warmup_schedule(optimizer, num_warmup_steps=args.bert_warmup_steps)
    scheduler_reducelr = get_reducelr_schedule(optimizer, patience=1)
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
            mlm_loss, itm_loss, mlm_accuracy, itm_accuracy = model(batch)
            mlm_loss = mlm_loss.mean() / accumulation_steps
            itm_loss = itm_loss.mean() / accumulation_steps
            mlm_accuracy = mlm_accuracy.mean()
            itm_accuracy = itm_accuracy.mean()
            loss = mlm_loss + itm_loss
            loss.backward() 
                
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler_warmup.step()
                
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
            
            if step % 2000 == 0:
                # 5. save checkpoint
                if loss.item() < loss_min:
                    loss_min = loss.item()
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'step': step, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss': itm_loss},
                            f'{args.savedmodel_path}/model_best.bin')
                    
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss': itm_loss},
                            f'{args.savedmodel_path}/model_best_epoch{epoch}.bin')

                    
 

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
