import torch
from utils.dataset import get_loader
import torch.nn.functional as F
import argparse
from model.sfmgtl import SFMGTL
import warnings
import numpy as np
import random
from torch.utils.data import DataLoader
import copy
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device('cuda:0')


def finetune_train_epoch(net_, adj1, adj2, adj3, loader_, optimizer_, mask=None):
    net_.train()
    epoch_loss = []
    for i, (x, y, t) in enumerate(loader_):
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        out, aux_loss = net_.evaluation(x, y, t, adj1, adj2, adj3)
        out = out.transpose(1, 2)[:, :, mask.view(-1).bool()]
        eff_batch_size = y.shape[0]
        y = y.view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
        loss = ((out - y) ** 2)
        loss = loss.mean(0).sum() + aux_loss * 0.5
        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=2)
        optimizer_.step()
        epoch_loss.append(loss.item())

    return epoch_loss


def train_epoch(net_, s_adj1, s_adj2, s_adj3,
                t_adj1, t_adj2, t_adj3,
                s_loader_, t_loader_, optimizer_, s_mask=None, t_mask=None):
    net_.train()
    epoch_loss = []
    epoch_acc = []
    epoch_ad = []
    if_reverse = True
    for i, (x, y, t) in enumerate(s_loader_):
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        p = float(i) / len(s_loader_)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        t_item = next(iter(t_loader_))
        t_x, t_y, t_t = t_item[0].to(device), t_item[1].to(device), t_item[2].to(device)
        s_pred, t_pred, s_aux_loss, t_aux_loss, acc, ad_loss = net_(x, y, t, s_adj1, s_adj2, s_adj3,
                                                                    t_x, t_y, t_t, t_adj1, t_adj2, t_adj3, alpha, if_reverse)

        if acc < 0.51:
            if_reverse = False
        else:
            if_reverse = True

        s_pred = s_pred.transpose(1, 2)[:, :, s_mask.view(-1).bool()]
        t_pred = t_pred.transpose(1, 2)[:, :, t_mask.view(-1).bool()]

        s_batch_size = y.shape[0]
        t_batch_size = t_y.shape[0]
        y = y.view(s_batch_size, 1, -1)[:, :, s_mask.view(-1).bool()]
        t_y = t_y.view(t_batch_size, 1, -1)[:, :, t_mask.view(-1).bool()]

        s_loss = ((s_pred - y) ** 2)
        s_loss = s_loss.mean(0).sum()
        t_loss = ((t_pred - t_y) ** 2)
        t_loss = t_loss.mean(0).sum()

        loss = s_loss + s_aux_loss + (t_loss + t_aux_loss) * 0.5 + ad_loss * 1

        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=2)
        optimizer_.step()
        epoch_loss.append((s_loss + s_aux_loss).item())
        epoch_acc.append(acc.item())
        epoch_ad.append((t_loss + t_aux_loss).item())

    return epoch_loss, epoch_acc, epoch_ad


def evaluate_epoch(net_, adj1, adj2, adj3, loader, spatial_mask):
    net_.eval()
    with torch.no_grad():
        se = 0
        ae = 0
        valid_points = 0
        for it_ in loader:
            (x, y, t) = it_
            x = x.to(device)
            y = y.to(device)
            t = t.to(device)
            out = net_.evaluation(x, y, t, adj1, adj2, adj3)[0].transpose(1, 2)[:, :, spatial_mask.view(-1).bool()]
            valid_points += x.shape[0] * spatial_mask.sum().item()
            out = torch.clip(out, min=0, max=None)
            batch_size = y.shape[0]
            lag = y.shape[1]
            y = y.view(batch_size, lag, -1)[:, :, spatial_mask.view(-1).bool()]
            se += ((out - y) ** 2).sum().item()
            ae += (out - y).abs().sum().item()
    return np.sqrt(se / valid_points), ae / valid_points


def train(args):
    s_dataset, _, _, s_mask, s_max, s_min, s_poi, \
    s_adj1, s_adj2, s_adj3 = get_loader(city=args.scity, type=args.datatype,
                                        used_day=args.train_days, pred_lag=args.pred_lag)

    t_loader, t_val_loader, t_test_loader, t_mask, t_max, t_min, t_poi, \
    t_adj1, t_adj2, t_adj3 = get_loader(city=args.tcity, type=args.datatype,
                                        used_day=args.train_days, pred_lag=args.pred_lag)
    net = SFMGTL(hidden_dim=args.hidden_dim).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print('Total params {} K'.format(total_params / 1000))
    optim = torch.optim.Adam([{'params': net.parameters()}], lr=1e-3)

    loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True)
    t_loader = DataLoader(t_loader, batch_size=args.batch_size, shuffle=True)

    source_loss_list = []
    target_loss_list = []

    if args.train_days == 1:
        all_epoch = 15
    else:
        all_epoch = 50

    for epoch in range(all_epoch):
        source_loss, acc_list, ad_list = train_epoch(net, s_adj1, s_adj2, s_adj3, t_adj1, t_adj2, t_adj3,
                                  loader, t_loader, optim, s_mask=s_mask, t_mask=t_mask)
        avg_source_loss = np.mean(source_loss)
        avg_acc = np.mean(acc_list)
        avg_ad = np.mean(ad_list)

        source_loss_list.append(avg_source_loss)
        target_loss_list.append(avg_ad)

        print('Epoch {}, loss {:.4}, acc {:.4}, ad {:.4}'.format(epoch,
                                                                 avg_source_loss,
                                                                 avg_acc,
                                                                 avg_ad))

    torch.save(net, 'model.pth')

    return net


def evaluation(args):
    t_loader, t_val_loader, t_test_loader, t_mask, t_max, t_min, t_poi, \
    t_adj1, t_adj2, t_adj3 = get_loader(city=args.tcity, type=args.datatype,
                                        used_day=args.train_days, pred_lag=args.pred_lag)
    net = torch.load('model.pth')
    for k, v in net.named_parameters():
        if k.split('.')[0] == 'common_attention':
            v.requires_grad = False
    net.node_knowledge.requires_grad = False
    net.zone_knowledge.requires_grad = False
    net.semantic_knowledge.requires_grad = False

    optim = torch.optim.Adam([{'params': net.parameters()}], lr=1e-3)
    t_loader = DataLoader(t_loader, batch_size=args.batch_size, shuffle=True)
    t_val_loader = DataLoader(t_val_loader, batch_size=args.batch_size, shuffle=False)
    t_test_loader = DataLoader(t_test_loader, batch_size=args.batch_size, shuffle=False)
    best_val_rmse = 1e10

    fine_loss_list = []

    for ep in range(100):
        # fine-tuning
        net.train()
        avg_loss = finetune_train_epoch(net, t_adj1, t_adj2, t_adj3, t_loader, optim, mask=t_mask)
        fine_loss_list.append(np.mean(avg_loss))
        print('Epoch %d, target pred loss %.4f' % (ep, np.mean(avg_loss)))
        net.eval()
        rmse_val, mae_val = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, t_val_loader, spatial_mask=t_mask)
        rmse_test, mae_test = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, t_test_loader, spatial_mask=t_mask)
        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            best_test_rmse = rmse_test
            best_test_mae = mae_test
            best_model = copy.deepcopy(net)
            print("Update best test...------------------------------------------------")
        print("validation rmse %.4f, mae %.4f" % (rmse_val * (t_max - t_min), mae_val * (t_max - t_min)))
        print("test rmse %.4f, mae %.4f" % (rmse_test * (t_max - t_min), mae_test * (t_max - t_min)))
        print()

    rmse_test, mae_test = evaluate_epoch(best_model, t_adj1, t_adj2, t_adj3, t_test_loader, spatial_mask=t_mask)
    print("Best test rmse %.4f, mae %.4f" % (rmse_test * (t_max - t_min), mae_test * (t_max - t_min)))
    torch.save(best_model, 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--scity', type=str, default='NY')
    parser.add_argument('--tcity', type=str, default='DC')
    parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--train_days', type=int, default=3)
    parser.add_argument('--pred_lag', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=32)
    args = parser.parse_args()

    setup_seed(args.seed)

    train(args)
    evaluation(args)
