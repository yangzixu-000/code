import os
import re
import random
import inspect
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_safe_name(model_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', model_name.lower())


def is_main_process():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        return local_rank == 0
    return True


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def create_optimizer(model, optimizer_config, use_layerwise_lr=True):
    opt_type = optimizer_config.get('type', 'AdamW').lower()
    base_lr = optimizer_config.get('lr', 1e-4)
    weight_decay = optimizer_config.get('weight_decay', 1e-5)
    betas = optimizer_config.get('betas', (0.9, 0.999))
    
    if use_layerwise_lr and hasattr(model, 'get_param_groups'):
        sig = inspect.signature(model.get_param_groups)
        params = {}
        if 'lr_scale_backbone' in sig.parameters:
            params['lr_scale_backbone'] = optimizer_config.get('lr_scale_backbone', 0.1)
        if 'lr_scale_head' in sig.parameters:
            params['lr_scale_head'] = optimizer_config.get('lr_scale_head', 1.0)
        if 'lr_scale_transformer' in sig.parameters:
            params['lr_scale_transformer'] = optimizer_config.get('lr_scale_transformer', 0.5)
        if 'lr_scale_cnn' in sig.parameters:
            params['lr_scale_cnn'] = optimizer_config.get('lr_scale_cnn', 0.1)
        
        param_groups = model.get_param_groups(**params)
        opt_params = []
        for group in param_groups:
            lr = base_lr * group.get('lr_scale', 1.0)
            opt_params.append({
                'params': group['params'],
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        if opt_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(opt_params, momentum=momentum)
        else:
            return optim.AdamW(opt_params, betas=betas)
    
    if opt_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=betas)


def create_scheduler(optimizer, config, steps_per_epoch=None):
    if config is None:
        return None
    sched_type = config.get('type', 'cosine').lower()
    per_batch = config.get('per_batch', False)

    if sched_type == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 20), gamma=config.get('gamma', 0.5))
    elif sched_type == 'cosine':
        T_max = config.get('T_max', 80)
        if per_batch and steps_per_epoch:
            T_max = T_max * steps_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.get('eta_min', 1e-7))
    else:
        T_0 = config.get('T_0', 10)
        if per_batch and steps_per_epoch:
            T_0 = T_0 * steps_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=config.get('T_mult', 2), eta_min=config.get('eta_min', 1e-6))
    scheduler.per_batch = per_batch
    return scheduler


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gather_tensors_with_lengths(tensor, dst=0):
    """
    跨卡聚合不等长张量。输入的 tensor 必须在 GPU 上。
    """
    if not is_distributed():
        return tensor
    world_size = get_world_size()
    rank = get_rank()
    local_len = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    all_lens = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(all_lens, local_len)
    all_lens = [int(l.item()) for l in all_lens]
    max_len = max(all_lens)
    if tensor.shape[0] < max_len:
        pad_size = max_len - tensor.shape[0]
        pad_shape = list(tensor.shape)
        pad_shape[0] = pad_size
        padded_tensor = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)], dim=0)
    else:
        padded_tensor = tensor
    gathered = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, padded_tensor)
    valid_tensors = [gathered[i][:length] for i, length in enumerate(all_lens)]
    if rank == dst:
        return torch.cat(valid_tensors, dim=0)
    else:
        return None


def all_reduce_metrics(*tensors, device=None):
    """
    跨卡求和指标，支持 GPU 张量。单卡直接返回标量。
    """
    if not is_distributed():
        return [t.item() if torch.is_tensor(t) else t for t in tensors]

    # 自动获取当前 GPU 设备
    if device is None:
        device = torch.device('cuda', get_rank())

    results = []
    for t in tensors:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        else:
            # 确保在正确的设备和 float32
            t = t.clone().detach().to(device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        results.append(t.item())
    return results


def train_model(model, model_name, train_loader, val_loader, test_loader,
                config, optimizer_config, scheduler_config=None):
    # 分布式初始化
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        local_rank = int(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if local_rank == 0:
            print(f"  🚀 Using DDP with {get_world_size()} GPUs")
    else:
        device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"  🚀 Using DataParallel with {num_gpus} GPUs")
            model = nn.DataParallel(model)
        else:
            if is_main_process():
                print("  💻 Using single GPU")

    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    os.makedirs(config['output']['results_path'], exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"{model_name}: {total_params:,} params, {trainable_params:,} trainable")

    use_layerwise_lr = config.get('use_layerwise_lr', True)
    optimizer = create_optimizer(model, optimizer_config, use_layerwise_lr)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, scheduler_config, steps_per_epoch)

    # 损失函数：标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    epochs = config['training']['epochs']
    model_save_path = config['output']['model_save_path']
    safe_name = get_safe_name(model_name)
    best_val_threshold = config.get('best_val_threshold', 0.1)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)

    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(epochs):
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs}', disable=not is_main_process())
        for batch_idx, (inputs, labels) in enumerate(pbar):
            if (labels == -1).any():
                valid_mask = labels != -1
                inputs = inputs[valid_mask]
                labels = labels[valid_mask]
                if inputs.size(0) == 0:
                    continue
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            if scheduler and scheduler.per_batch:
                scheduler.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            if batch_idx % 20 == 0 and is_main_process():
                current_acc = 100. * preds.eq(labels).sum().item() / labels.size(0)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})

        if scheduler and not scheduler.per_batch:
            scheduler.step()

        # ===== 训练指标聚合（使用 device） =====
        if is_distributed():
            train_loss, train_correct, train_total = all_reduce_metrics(
                train_loss, train_correct, train_total, device=device
            )
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if scaler:
                    with autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        # ===== 验证指标聚合（使用 device） =====
        if is_distributed():
            val_total, val_correct = all_reduce_metrics(val_total, val_correct, device=device)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process():
            lr_str = f'{current_lr:.2e}' if current_lr < 1e-4 else f'{current_lr:.6f}'
            print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {lr_str}')

        train_losses.append(train_loss / len(train_loader) if train_total > 0 else 0)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc + best_val_threshold:
            best_val_acc = val_acc
            raw_model = model.module if hasattr(model, 'module') else model
            model_filename = f"best_{safe_name}.pth"
            torch.save(raw_model.state_dict(), os.path.join(model_save_path, model_filename))
            if is_main_process():
                print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")

    # 测试
    model_path = os.path.join(model_save_path, f"best_{safe_name}.pth")
    raw_model = model.module if hasattr(model, 'module') else model
    if os.path.exists(model_path):
        raw_model.load_state_dict(torch.load(model_path, map_location=device))
        if is_main_process():
            print(f"Loaded best model for testing: {model_path}")
    else:
        if is_main_process():
            print(f"Warning: Best model not found at {model_path}, using current model for testing")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', disable=not is_main_process()):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            _, preds = outputs.max(1)
            # 保留 GPU 张量，不转 cpu
            all_preds.append(preds)
            all_labels.append(labels)

    test_acc = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if all_preds:
        # 合并 GPU 张量
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        
        if is_distributed():
            # gather 在 GPU 上完成
            gathered_preds = gather_tensors_with_lengths(preds_tensor, dst=0)
            gathered_labels = gather_tensors_with_lengths(labels_tensor, dst=0)
            
            if is_main_process() and gathered_preds is not None:
                preds_np = gathered_preds.cpu().numpy()
                labels_np = gathered_labels.cpu().numpy()
            else:
                preds_np = np.array([])
                labels_np = np.array([])
        else:
            preds_np = preds_tensor.cpu().numpy()
            labels_np = labels_tensor.cpu().numpy()

        if is_main_process() and len(preds_np) > 0:
            test_acc = 100.0 * (preds_np == labels_np).sum() / len(preds_np)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds_np, average='weighted', zero_division=0
            )
            print(f"Test Acc: {test_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%, F1: {f1:.4f}")

    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()

    return {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs
    }