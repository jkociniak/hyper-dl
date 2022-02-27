import torch
from collections import defaultdict


def train_loop(run, epochs, model, criterion, metrics, optimizer, scheduler, train_loader, val_loader, test_loader):
    device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    for epoch in range(epochs):
        train_loss, train_metrics = train(run, model, criterion, metrics, optimizer, train_loader, device)
        print(f'[{epoch + 1}] train loss (mean of batch losses): {train_loss}')

        val_loss, val_metrics = evaluate(run, model, criterion, metrics, val_loader, device, 'val')
        print(f'[{epoch + 1}] val loss: {val_loss}')

        if scheduler is not None:
            scheduler.step(val_loss)

    print('Training finished')

    test_loss, test_metrics = evaluate(run, model, criterion, metrics, test_loader, device, 'test')
    print(f'test loss: {test_loss}')
    print('Experiment finished')

    return test_loss, test_metrics


def train(run, model, criterion, metrics_dict, optimizer, loader, device):
    model.train()

    loss = 0
    metrics = defaultdict(float)
    n_samples = 0

    for i, data in enumerate(loader):
        pairs, dist = data['pairs'], data['dist']
        x1, x2 = pairs[:, 0, :], pairs[:, 1, :]
        x1, x2 = x1.to(device), x2.to(device)
        n_samples += x1.size(dim=0)

        optimizer.zero_grad()
        dist_pred = model(x1, x2)

        batch_loss = criterion(dist_pred, dist)
        batch_loss.backward()
        optimizer.step()

        batch_loss *= loader.batch_size
        loss += batch_loss.item()

        for name, metric in metrics_dict.items():
            metrics[name] += metric(dist_pred, dist).item() * loader.batch_size

    loss /= n_samples
    metrics = {name: val / n_samples for name, val in metrics.items()}

    run['metrics/train/loss'].log(loss)
    for metric_name, metric_val in metrics.items():
        run[f'metrics/train/{metric_name}'].log(metric_val)

    return loss, metrics


def evaluate(run, model, criterion, metrics_dict, loader, device, set_name):
    model.eval()

    loss = 0
    metrics = defaultdict(float)
    n_samples = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            pairs, dist = data['pairs'], data['dist']
            x1, x2 = pairs[:, 0, :], pairs[:, 1, :]
            x1, x2 = x1.to(device), x2.to(device)
            n_samples += x1.size(dim=0)

            dist_pred = model(x1, x2)

            batch_loss = criterion(dist_pred, dist).item() * loader.batch_size
            loss += batch_loss

            for name, metric in metrics_dict.items():
                metrics[name] += metric(dist_pred, dist).item() * loader.batch_size

    loss /= n_samples
    metrics = {name: val / n_samples for name, val in metrics.items()}

    run[f'metrics/{set_name}/loss'].log(loss)
    for metric_name, metric_val in metrics.items():
        run[f'metrics/{set_name}/{metric_name}'].log(metric_val)

    return loss, metrics