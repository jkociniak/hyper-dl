import torch
from collections import defaultdict


def train_loop(run, epochs, model, criterion, metrics_dict, optimizer, scheduler, train_loader, val_loader):
    device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)
    last_val_metrics = None

    for epoch in range(epochs):
        train_loss, train_metrics = train(model, criterion, metrics_dict, optimizer, train_loader, device)
        run['metrics/train/loss'].log(train_loss)
        for metric_name, metric_val in train_metrics.items():
            run[f'metrics/train/{metric_name}'].log(metric_val)
        print(f'[{epoch + 1}] train loss: {train_loss}')

        val_loss, val_metrics = evaluate(model, criterion, metrics_dict, val_loader, device)
        run['metrics/val/loss'].log(val_loss)
        for metric_name, metric_val in val_metrics.items():
            run[f'metrics/val/{metric_name}'].log(metric_val)
        print(f'[{epoch + 1}] val loss: {val_loss}')

        if scheduler is not None:
            scheduler.step(val_loss)

        last_val_metrics = val_loss, val_metrics

    print('Training finished')
    return last_val_metrics


def train(model, criterion, metrics_dict, optimizer, loader, device):
    model.train()

    losses = []
    metrics = defaultdict(list)

    for i, data in enumerate(loader):
        pairs, dist = data['pairs'], data['dist']
        x1, x2 = pairs[:, 0, :], pairs[:, 1, :]
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()
        dist_pred = model(x1, x2)

        loss = criterion(dist_pred, dist)
        losses.append(loss)

        for name, metric in metrics_dict.items():
            metrics[name] = metric(dist_pred, dist).item()

        loss.backward()
        optimizer.step()

    loss = torch.tensor(losses).mean()
    metrics = {metric_name: torch.tensor(metric_val).mean()
               for metric_name, metric_val in metrics.items()}

    return loss, metrics


def evaluate(model, criterion, metrics_dict, loader, device):
    model.eval()

    losses = []
    metrics = defaultdict(list)

    with torch.no_grad():
        for i, data in enumerate(loader):
            pairs, dist = data['pairs'], data['dist']
            x1, x2 = pairs[:, 0, :], pairs[:, 1, :]
            x1, x2 = x1.to(device), x2.to(device)

            dist_pred = model(x1, x2)

            loss = criterion(dist_pred, dist)
            losses.append(loss.item())
            for name, metric in metrics_dict.items():
                metrics[name].append(metric(dist_pred, dist).item())

    loss = torch.tensor(losses).mean()
    metrics = {metric_name: torch.tensor(metric_val).mean()
               for metric_name, metric_val in metrics.items()}

    return loss, metrics