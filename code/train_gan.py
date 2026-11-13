import datetime, os, random, uuid

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from option import args, CKPT_DIR, LOG_DIR
import datasets as dataset
from utils import logger_config
from model.wllm import wllm

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt} ({percentage:3.0f}%) {postfix} > Elapsed: {elapsed}, Remaining: {remaining}"


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_deterministic(args.seed)


def criterion(model, pred_dict, label, config):
    logits = pred_dict['logits']
    loss_bce = F.binary_cross_entropy_with_logits(
        logits, label, reduction='none')

    loss_classify = torch.mean(loss_bce)
    smoothing = 0.1
    targets_smooth = label * (1 - smoothing) + 0.5 * smoothing
    loss_smooth = F.binary_cross_entropy_with_logits(logits, targets_smooth)
    loss_smooth = torch.mean(loss_smooth) * config.loss_smooth
    llm_out_src = pred_dict['llm_out_src']
    llm_out_his = pred_dict['llm_out_his']
    llm_out_residual = pred_dict['llm_out_residual']
    loss_residual = model.residual_history.loss_fn(llm_out_residual, llm_out_src, llm_out_his)
    loss = config.lambda_1 * loss_classify + \
           config.lambda_2 * loss_smooth + \
           config.lambda_3 * loss_residual['total_loss']

    return loss, loss_classify, loss_smooth, loss_residual['total_loss']


def eval_epoch(dataset, model, device, config):
    m_pred, m_label = [], []
    with torch.no_grad():
        model.eval()
        with tqdm(total=len(dataset), bar_format=BAR_FORMAT,
                  desc="Eval:") as t:
            for batch_sample in dataset:
                x = model.infer_forward(batch_sample, device)

                y = batch_sample['labels'].to(device)
                pred_score = x['logits'].sigmoid().cpu().numpy().flatten()

                y = y.cpu().numpy().flatten()
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y))

                auc_roc = roc_auc_score(m_label, m_pred)
                t.set_postfix(roc_auc=np.mean(auc_roc))
                t.update(1)

    auc_roc = roc_auc_score(m_label, m_pred)
    ap = average_precision_score(m_label, m_pred)
    return auc_roc, (m_label, m_pred), ap


def get_checkpoint_path(model_name: str = "sad", dataset_name: str = "wikipedia", auc: float = 0.5):
    save_path = os.path.join(CKPT_DIR, "train_weight")
    f_auc = "_".join(str(auc).split("."))

    ret = f"{save_path}/{model_name}-{dataset_name}-{f_auc}-node-classification.pth"
    os.makedirs(save_path, exist_ok=True)
    return ret


def save_label_nn_out(save_model_path: str, label: torch.Tensor, nn_out: torch.Tensor, dt="val") -> None:
    out_path = save_model_path.replace(".pth", f"-label_pred-{dt}.pth")
    torch.save({
        "label": label,
        "nn_out": nn_out,
    }, out_path)


def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict()
    }, path)

DATA = args.data_set
MODE = args.mode
Prefix = f"{MODE}_{DATA}"

config = args
device = config.device

now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(LOG_DIR, exist_ok=True)
logger = logger_config(log_path=os.path.join(LOG_DIR, f'{MODE}-{DATA}-{now_time}.log'), logging_name=MODE)
logger.info(config)

dataset_train = dataset.DygDataset(config, 'train')
dataset_valid = dataset.DygDataset(config, 'valid')
dataset_test = dataset.DygDataset(config, 'test')

config.n_nodes = dataset_train.num_nodes
collate_fn = dataset.Collate(config)

model = wllm(config)
model = model.to(device)

trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

optimizer = torch.optim.AdamW(trained_parameters, lr=config.learning_rate)
scheduler = StepLR(optimizer, step_size=config.learning_rate_drop_step, gamma=config.learning_rate_gamma)

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
)

loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
)

loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
)

max_val_auc, max_test_auc = 0.0, 0.0
last_val_auc, last_test_auc = 0.0, 0.0
best_auc = [0, 0, 0]
step_flag = True

if __name__ == '__main__':
    for epoch in range(config.n_epochs):
        m_loss = []
        loss_a_list = []
        loss_b_list = []
        loss_c_list = []
        loss_d_list = []
        loss_class_list = []
        loss_class_smooth_list = []
        with tqdm(total=len(loader_train), bar_format=BAR_FORMAT, desc=f"Train {epoch}:") as t:
            model.train()
            model.memory_pool.init_memory()
            model.history_queue.clear()
            for batch_sample in loader_train:
                t.set_description('Train Epoch %i' % epoch)

                pred_x = model.train_forward(batch_sample, device)
                label_y = batch_sample['labels'].to(device)

                loss, loss_classify, loss_smooth, loss_residual = criterion(model, pred_x, label_y, config)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
                optimizer.step()

                with torch.no_grad():
                    m_loss.append(loss.item())
                    loss_class_list.append(loss_classify.item())
                    loss_class_smooth_list.append(loss_smooth.item())
                    loss_a_list.append(loss_residual.item())

                    t.set_postfix(loss=np.mean(m_loss),
                                  loss_class=np.mean(loss_class_list),
                                  loss_smooth=np.mean(loss_class_smooth_list),
                                  loss_residual=np.mean(loss_a_list)
                                  )
                    t.update(1)
            scheduler.step()

        val_auc, (val_label, val_pred), val_ap = eval_epoch(loader_valid, model, device, config)
        best_model_path = get_checkpoint_path(model_name=Prefix + str(uuid.uuid4()), dataset_name=DATA, auc=0.1)
        save_model(model, best_model_path)
        test_auc, (test_label, test_pred), test_ap = eval_epoch(loader_test, model, device, config)

        last_val_auc, last_test_auc = val_auc, test_auc
        if val_auc > best_auc[1]:
            best_auc = [epoch, val_auc, test_auc]

        logger.info(f"=========================== epoch: {epoch} ===========================")
        logger.info(
            f'train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, '
            f'smooth class loss: {np.mean(loss_class_smooth_list)} '
            f'residual loss: {np.mean(loss_a_list)} ')
        logger.info('<<<<< val auc roc:{}, val auc auc pr:{} >>>>>'.format(val_auc, val_ap))
        logger.info('>>>>> test auc roc:{}, test auc pr:{} <<<<<'.format(test_auc, test_ap))
        logger.info(f"=========================== epoch: {epoch} ===========================")

        if epoch < config.learning_rate_drop_step:
            if val_auc >= config.save_auc or (config.save_auc_rf and val_auc <= config.save_auc):
                logger.info(f"+++++++++++++++++++++++ best epoch: {epoch}, val auc: {val_auc} +++++++++++++++++++++++")
                if step_flag:
                    for _ in range(config.learning_rate_drop_step - epoch + 1):
                        scheduler.step()
                else:
                    step_flag = False

        if not step_flag:
            new_best_model_path = get_checkpoint_path(model_name=Prefix + str(epoch), dataset_name=DATA, auc=test_auc)
            if test_auc > max_test_auc:
                os.rename(best_model_path, new_best_model_path)
            else:
                os.remove(best_model_path)
            save_label_nn_out(new_best_model_path, test_label, test_pred, dt="test")
        else:
            os.remove(best_model_path)

        max_val_auc, max_test_auc = max(max_val_auc, val_auc), max(max_test_auc, test_auc)

    logger.info(f"$$$$$$$$$$$$ last valid auc: val={last_val_auc}, test={last_test_auc} $$$$$$$$$$$$")
    logger.info(f"$$$$$$$$$$$$ best valid auc: epoch={best_auc[0]}, val={best_auc[1]}, test={best_auc[2]} $$$$$$$$$$$$")
    logger.info(f"XXXXXXXXXXXX max valid auc: {max_val_auc}, max test auc: {max_test_auc} XXXXXXXXXXXX")
