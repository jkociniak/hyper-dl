import torch
import pandas as pd
from collections import defaultdict
from neptune.new.types import File


def train_loop(run, epochs, model, criterion, metrics, optimizer, scheduler, r_optimizer, r_scheduler, loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Found device: {device}')
    model = torch.nn.DataParallel(model)
    model.to(device)

    for epoch in range(epochs):
        train_loss = train(run, model, criterion, metrics, optimizer, r_optimizer, loaders['train'], device)
        print(f'[{epoch + 1}] train loss (mean of batch losses): {train_loss}')

        val_loss = evaluate(run, model, criterion, metrics, loaders['val'], device)
        print(f'[{epoch + 1}] val loss: {val_loss}')

        if scheduler is not None:
            scheduler.step(val_loss)

        if r_scheduler is not None:
            r_scheduler.step(val_loss)

    print('Training finished')

    print('Generating results...')
    results = final_evaluate(run, model, criterion, metrics, loaders, device)

    print('Experiment finished')

    return results


def train(run, model, criterion, metrics_dict, optimizer, r_optimizer, loader, device):
    model.train()

    loss = 0
    metrics = defaultdict(float)
    n_samples = 0

    for i, data in enumerate(loader):
        pairs, dist = data['pairs'], data['dist']
        pairs, dist = pairs.to(device), dist.to(device)
        x1, x2 = pairs[:, 0, :], pairs[:, 1, :]

        n_samples += x1.size(dim=0)

        optimizer.zero_grad()
        dist_pred = model(x1, x2)
        dist_pred = dist_pred.squeeze()

        batch_loss = criterion(dist_pred, dist)
        batch_loss.backward()

        if optimizer is not None:
            optimizer.step()
        if r_optimizer is not None:
            r_optimizer.step()

        batch_loss *= loader.batch_size
        loss += batch_loss.item()

        for name, metric in metrics_dict.items():
            metrics[name] += metric(dist_pred, dist).item() * loader.batch_size

    loss /= n_samples
    metrics = {name: val / n_samples for name, val in metrics.items()}

    if run is not None:
        run['metrics/train/loss'].log(loss)
        for metric_name, metric_val in metrics.items():
            run[f'metrics/train/{metric_name}'].log(metric_val)

    return loss


def evaluate(run, model, criterion, metrics_dict, loader, device):
    model.eval()

    loss = 0
    metrics = defaultdict(float)
    n_samples = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            pairs, dist = data['pairs'], data['dist']
            pairs, dist = pairs.to(device), dist.to(device)
            x1, x2 = pairs[:, 0, :], pairs[:, 1, :]
            n_samples += x1.size(dim=0)

            dist_pred = model(x1, x2)
            dist_pred = dist_pred.squeeze()

            batch_loss = criterion(dist_pred, dist).item() * loader.batch_size
            loss += batch_loss

            for name, metric in metrics_dict.items():
                metrics[name] += metric(dist_pred, dist).item() * loader.batch_size

    loss /= n_samples
    metrics = {name: val / n_samples for name, val in metrics.items()}

    if run is not None:
        run[f'metrics/val/loss'].log(loss)
        for metric_name, metric_val in metrics.items():
            run[f'metrics/val/{metric_name}'].log(metric_val)

    return loss


def final_evaluate(run, model, criterion, metrics_dict, loaders, device):
    model.eval()

    results = {}
    for name, loader in loaders.items():
        dists = []
        preds = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                pairs, dist = data['pairs'], data['dist']
                pairs, dist = pairs.to(device), dist.to(device)
                x1, x2 = pairs[:, 0, :], pairs[:, 1, :]

                dist_pred = model(x1, x2)
                dist_pred = dist_pred.squeeze()

                dists.append(dist)
                preds.append(dist_pred)

        dists = torch.cat(dists)
        preds = torch.cat(preds)
        metrics = {}

        for metric_name, metric in metrics_dict.items():
            metrics[metric_name] = metric(preds, dists, reduction='none')
            metrics[metric_name] = metrics[metric_name].cpu()

        dists = dists.cpu()
        preds = preds.cpu()
        results_df = pd.DataFrame({
            'dist': dists,
            'pred': preds,
            **metrics
        })

        results[name] = results_df
        # n_samples = results_df.shape[0]
        # if run is not None and n_samples < :
        #     run[f'results/{name}/res_table'].upload(File.as_html(results_df))

    return results


# def after_eval(model):
#     cl = model.encoder.concat_layer
#     w1 = cl.mfc1.fc.weight
#     b1 = cl.mfc1.fc.bias
#     w2 = cl.mfc2.fc.weight
#     b2 = cl.mfc2.fc.bias
#     print()
#     print(f'W1: {w1}')
#     print(f'W1^T W1 = {w1 @ w1.T}')
#     print(f'b1: {b1}')
#     print(f'W2: {w2}')
#     print(f'W1^T W1 = {w2 @ w2.T}')
#     print(f'b2: {b2}')
#
#
# def add_artifacts(model):
#     for idx, m in enumerate(model.named_modules()):
#         print(idx, '->', m)
