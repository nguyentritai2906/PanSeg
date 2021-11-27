import torch
from config import config
from data.build import build_train_loader_from_cfg
from model.models import FPSNet
from solver.build import build_lr_scheduler, build_optimizer
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from utils.utils import to_cuda


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FPSNet(19)
    model = model.to(device)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    data_loader = build_train_loader_from_cfg(config)
    data_loader_iter = iter(data_loader)
    scaler = GradScaler(enabled=config.TRAIN.AMP)
    try:
        for i in range(1):
            # data
            data = next(data_loader_iter)
            # ['raw_size', 'size', 'image', 'semantic', 'foreground', 'center',
            # 'center_points', 'offset', 'semantic_weights', 'center_weights',
            # 'offset_weights']
            data = to_cuda(data, device)
            print(type(data[0]))

            image = data['image']
            with autocast(enabled=config.TRAIN.AMP):
                out_dict = model(image, data)
            loss = out_dict['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
    except Exception:
        raise


if __name__ == "__main__":
    main()
