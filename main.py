import argparse
import os
import numpy as np
import time
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
import torch
import torch.nn as nn
from collections import Counter
from datasets.mydataloader import Deception_Dataset
import torch.utils.data as data
from models.DLF_BRAM import DLF_BRAM_NET
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.backends.cudnn.enabled = True
import random


def parse_options():
    parser = argparse.ArgumentParser(description="DLF-BRAM Deception Detection")

    # Train Parameter
    parser.add_argument('--device', type=str, default="cuda:0", help='GPU device id')
    parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate')
    parser.add_argument('--when', type=int, default=15, help='When to decay learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_worker', type=int, default=16, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')

    # Model Parameter
    parser.add_argument('--blocks', type=int, default=5, help='Number of patch blocks')
    parser.add_argument('--len', type=int, default=4, help='Temporal length (frames)')
    parser.add_argument('--depth', type=int, default=4, help='Transformer depth')
    parser.add_argument('--size', type=int, default=96, help='Input image size')

    # Data Parameter
    parser.add_argument('--data_name', type=str, default='Dolos', help='Dataset name (folder prefix)')
    parser.add_argument('--train_flag', type=int, default=1, help='Data split mode: 1/2/3 for different folds')
    parser.add_argument('--frame_root', type=str, default='./data', help='Root folder containing frames and annotations')
    parser.add_argument('--save_model_dir', type=str, default='./checkpoints', help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='log', help='Directory to store logs')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pretrained model for testing')

    # Model_Version
    parser.add_argument('--m_version', type=str, default='DLF_BRAM', help='Model version tag')

    parser.add_argument('--test', action='store_true', help='Only run inference on test set')


    opts = parser.parse_args()

    # set_seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    opts.device = torch.device(opts.device)

    # logfile path
    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.save_model_dir, exist_ok=True)

    return opts


def cal_leability(output,label):
    output = np.asarray(output)
    assert output.shape[1] == 2,"output error"
    label = np.asarray(label).flatten()

    output_label = np.argmax(output,axis=1)
    output_label = np.asarray(output_label).flatten()
    result = np.abs(output[:, 0] - output[:, 1])
    result = np.asarray(result).flatten()
    assert len(output_label) == len(label),"length error"

    lea_ablity  = 0
    for i in range(len(label)):
        if(label[i]==output_label[i]):
            lea_ablity = lea_ablity + result[i] #
        else:
            lea_ablity = lea_ablity - result[i] #
    return lea_ablity


def update(w1,w2,w3):
    W = np.asarray([w1,w2,w3])
    sum_W = np.sum(W, axis=0)
    W = 1-W/sum_W
    return W

def update_val(w1,w2,w3):
    W = np.asarray([w1,w2,w3])

    sum_W = np.sum(W, axis=0)

    W = W/sum_W
    return W

def val_one_epoch(args, val_data_loader, model, loss_fn, W):

    epoch_loss = []
    epoch_loss_1 = []
    epoch_loss_2 = [] 
    epoch_loss_3 = []

    epoch_predictions = []
    epoch_predictions_1 = []
    epoch_predictions_2 = []
    epoch_predictions_3 = []
    epoch_labels = []

    start_time = time.time()
    vid = []
    model.eval()

    with torch.no_grad():
        for imgs, label, vid_name in val_data_loader:

            imgs = imgs.to(args.device)

            vid.append(list(vid_name))
            label = label.to(args.device)

            # Forward
            p1, p2, p3 = model(imgs)

            p = W[0]*p1 + W[1]*p2 + W[2]*p3

            _loss_1 = loss_fn(p1, label)
            loss_1 = _loss_1.item()
            epoch_loss_1.append(loss_1)

            _loss_2 = loss_fn(p2, label)
            loss_2 = _loss_2.item()
            epoch_loss_2.append(loss_2)

            _loss_3 = loss_fn(p3, label)
            loss_3 = _loss_3.item()
            epoch_loss_3.append(loss_3)

            _loss = loss_fn(p, label)
            loss = _loss.item()
            epoch_loss.append(loss)

            epoch_predictions_1.append(torch.argmax(p1, dim=1))
            epoch_predictions_2.append(torch.argmax(p2, dim=1))
            epoch_predictions_3.append(torch.argmax(p3, dim=1))
            epoch_predictions.append(torch.argmax(p, dim=1))

            epoch_labels.append(label)

    vid = np.asarray(vid).flatten()

    epoch_predictions = torch.cat(epoch_predictions)
    epoch_predictions_1 = torch.cat(epoch_predictions_1)
    epoch_predictions_2 = torch.cat(epoch_predictions_2)
    epoch_predictions_3 = torch.cat(epoch_predictions_3)
    epoch_labels = torch.cat(epoch_labels)

    end_time = time.time()
    total_time = end_time - start_time
    print('val_total_time:', total_time)

    epoch_loss_1 = np.mean(epoch_loss_1)
    epoch_loss_2 = np.mean(epoch_loss_2)
    epoch_loss_3 = np.mean(epoch_loss_3)
    epoch_loss = np.mean(epoch_loss)

    return epoch_loss,epoch_loss_1,epoch_loss_2,epoch_loss_3, epoch_predictions,epoch_predictions_1,epoch_predictions_2,epoch_predictions_3, epoch_labels, vid



def train_one_epoch(args, train_data_loader, model,optimizer,loss_fn,W):
    ### Local Parameters
    epoch_loss = []
    epoch_predictionsc = []
    epoch_predictionsc1 = []
    epoch_predictionsc2 = []
    epoch_predictionsc3 = []
    epoch_labels = []
    start_time = time.time()

    model.train()
    for cur_iter, (img, label,vid_name) in enumerate(train_data_loader):
        # Loading waves and labels to device
        img = img.to(args.device)
        label = label.to(args.device)
        epoch_labels.append(label)
        # Reseting Gradients
        optimizer.zero_grad()

        # Forward
        p1,p2,p3 = model(img)

        P = p1 + p2 + p3

        epoch_predictionsc1.append(p1)
        epoch_predictionsc2.append(p2)
        epoch_predictionsc3.append(p3)
        epoch_predictionsc.append(torch.argmax(P, dim=1))

        loss1 = loss_fn(p1, label)
        loss2 = loss_fn(p2, label)
        loss3 = loss_fn(p3, label)
        loss = W[0]*loss1+ W[1]*loss2+ W[2]*loss3  

        epoch_loss.append(loss.item())
        # Backward
        loss.backward()
        optimizer.step()

        if cur_iter % 50 == 0:
            print("iter {}, loss {:.20f}".format(str(cur_iter), loss))

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    print("total_time:" , total_time)
    epoch_predictionsc = torch.cat(epoch_predictionsc)
    epoch_predictionsc1 = torch.cat(epoch_predictionsc1)
    epoch_predictionsc2 = torch.cat(epoch_predictionsc2)
    epoch_predictionsc3 = torch.cat(epoch_predictionsc3)
    epoch_labels = torch.cat(epoch_labels)

    return epoch_predictionsc,epoch_predictionsc1,epoch_predictionsc2,epoch_predictionsc3,epoch_labels,epoch_loss


def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score

def eval_vid(vid_name, pred, data_name='Dolos'):
    res = {}

    assert len(vid_name) == len(pred), "shape error"
    for i in range(len(vid_name)):
        vid = vid_name[i]
        pr = pred[i]

        if vid not in res:
            res[vid] = [pr]
        else:

            res[vid].append(pr)
    vid_label = []
    vid_pred = []
    if data_name != 'MU3D':
        for key, value in res.items():
            if ("lie" in key):
                vid_label.append(1)
            elif ("tru" in key):
                vid_label.append(0)
            else:
                print("label error")
            element_count = Counter(value)
            total = sum(element_count.values())

            most_common_element = element_count.most_common(1)

            vid_pred.append(most_common_element[0][0])
    else:
        for key, value in res.items():
            if ("L" in key):
                vid_label.append(1)
            elif ("T" in key):
                vid_label.append(0)
            else:
                print("label error")
            element_count = Counter(value)
            total = sum(element_count.values())

            most_common_element = element_count.most_common(1)

            vid_pred.append(most_common_element[0][0])


    acc, f1, auc_score = evaluation(vid_label, vid_pred)

    return acc, f1, auc_score



def train_test(log_name,filename,args):

    if args.blocks == 5:
        keyblockdir = 'keyblock7head'
    elif args.blocks == 7:
        keyblockdir = 'keyblock7head'
    else:
        keyblockdir = 'keyblock'
    file_folder = os.path.join(args.frame_root, args.data_name, 'frames')
    annotations_folder = os.path.join(args.frame_root, args.data_name, keyblockdir)

    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    f = open(log_name, 'a')
    f.write(filename)
    f.write('\n')

    savepath = os.path.join(args.save_model_dir, filename)
    os.makedirs(savepath, exist_ok=True)

    train_dataset = Deception_Dataset(file_folder, annotations_folder,args.data_name, args.train_flag, args.size,args.len, args.blocks)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, drop_last=True)

    test_dataset = Deception_Dataset(file_folder, annotations_folder,args.data_name, args.train_flag,args.size,args.len, args.blocks, train = False)

    test_batch = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=16, drop_last=True)

    model = DLF_BRAM_NET(img_size=args.size, num_classes=2, num_patches= args.blocks, num_frames=args.len, depth = args.depth)
    print(args.m_version, args.blocks)

    model.to(args.device)
    print(model)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
    best_acc = 0.0
    count = 0
    best_model_wts = model.state_dict()
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    print("\t Started Training")
    # epoch_dict={}
    W = [1,1,1]
    W_val = [1/3,1/3,1/3]
    for epoch in range(args.num_epochs):
        cur_epoch = epoch + 1
        print('\t\t Epoch....', epoch + 1)
        print(args.lr)

        print("W:",W)

        preds, preds_1,preds_2,preds_3,epoch_labels,epoch_loss= train_one_epoch(args, train_batch, model, optimizer,loss_fn, W)
        train_acc, train_f1, train_auc = evaluation(epoch_labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

        val_loss,val_loss_1,val_loss_2,val_loss_3, val_preds,val_preds_1,val_preds_2,val_preds_3, val_lables, vid = val_one_epoch(args, test_batch, model, loss_fn, W_val)

        val_acc_1, val_f1_1, val_auc_1 = eval_vid(vid, val_preds_1.detach().cpu().numpy(), args.data_name)
        val_acc_2, val_f1_2, val_auc_2 = eval_vid(vid, val_preds_2.detach().cpu().numpy(), args.data_name)
        val_acc_3, val_f1_3, val_auc_3 = eval_vid(vid, val_preds_3.detach().cpu().numpy(), args.data_name)
        val_acc, val_f1, val_auc = eval_vid(vid, val_preds.detach().cpu().numpy(), args.data_name)


        w1 = cal_leability(preds_1.detach().cpu().numpy(),epoch_labels.cpu().numpy())
        w2 = cal_leability(preds_2.detach().cpu().numpy(),epoch_labels.cpu().numpy())
        w3 = cal_leability(preds_3.detach().cpu().numpy(),epoch_labels.cpu().numpy())

        W = update(w1,w2,w3)
        # W_val = update_val(w1,w2,w3)

        print("epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
              "test_acc_1 {:.5f}, test_f1_1: {:.5f}, test_auc_1:{:.5f}".format(cur_epoch, train_acc, train_f1, train_auc,
                                                                         val_acc_1, val_f1_1, val_auc_1))
        print("epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
              "test_acc_2 {:.5f}, test_f1_2: {:.5f}, test_auc_2:{:.5f}".format(cur_epoch, train_acc, train_f1, train_auc,
                                                                         val_acc_2, val_f1_2, val_auc_2))
        print("epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
              "test_acc_3 {:.5f}, test_f1_3: {:.5f}, test_auc_3:{:.5f}".format(cur_epoch, train_acc, train_f1, train_auc,
                                                                         val_acc_3, val_f1_3, val_auc_3))

        print("epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
              "ALL_test_acc {:.5f}, ALL_test_f1: {:.5f}, ALL_test_auc:{:.5f}".format(cur_epoch, train_acc, train_f1, train_auc,
                                                                         val_acc, val_f1, val_auc))
        f.write(
            "epoch {}, test_acc {:.5f}, test_f1: {:.5f}, test_auc:{:.5f}".format(cur_epoch, val_acc, val_f1, val_auc))
        f.write("\n")
        if val_acc == best_acc:
            count += 1
        if val_acc > best_acc:
            count = 0
        if val_acc >= best_acc:
            # 保存最好结果最新模型
            best_acc = val_acc
            best_model_wts = model.state_dict()
            results = "best results are acc {:.5f}, f1: {:.5f}, auc:{:.5f} ".format(val_acc, val_f1, val_auc)
            val_results = classification_report(val_lables.cpu().numpy(), val_preds.cpu().numpy(),
                                                target_names=["truth", "deception"])

    torch.save(best_model_wts, os.path.join(savepath, "bestepoch.pth"))

    f.write("****************\n")
    f.write(results)
    f.write("\n\n")
    f.write(val_results)
    f.write("\n\n")

    f.close()

def only_test(log_name,filename,args):

    # 根据块数判断注释目录
    if args.blocks in [5, 7]:
        keyblockdir = 'keyblock7head'
    else:
        keyblockdir = 'keyblock'

    file_folder = os.path.join(args.frame_root, args.data_name, 'frames')
    annotations_folder = os.path.join(args.frame_root, args.data_name, keyblockdir)

    if not os.path.exists(file_folder):
        raise FileNotFoundError(f"Frame path not found: {file_folder}")
    if not os.path.exists(annotations_folder):
        raise FileNotFoundError(f"Annotation path not found: {annotations_folder}")

    print(f"[INFO] Loading test data from:\n - frames: {file_folder}\n - annotations: {annotations_folder}")


    test_dataset = Deception_Dataset(file_folder, annotations_folder,args.data_name, 
                              args.train_flag,args.size,args.len, args.blocks, train = False)

    test_batch = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=16, drop_last=True)

    model = DLF_BRAM_NET(img_size=args.size, num_classes=2, num_patches= args.blocks, num_frames=args.len, depth = args.depth)

    print(f"[INFO] Loading model weights from: {args.pretrained_path}")
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"Model file not found: {args.pretrained_path}")
    state_dict = torch.load(args.pretrained_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    print("[INFO] Model ready. Running inference...")

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    print("\t Started Training")

    val_loss,val_loss_1,val_loss_2,val_loss_3, val_preds,val_preds_1,val_preds_2,val_preds_3, val_lables, vid = val_one_epoch(args, test_batch, model, loss_fn)

    val_acc_1, val_f1_1, val_auc_1 = eval_vid(vid, val_preds_1.detach().cpu().numpy(), args.data_name)
    val_acc_2, val_f1_2, val_auc_2 = eval_vid(vid, val_preds_2.detach().cpu().numpy(), args.data_name)
    val_acc_3, val_f1_3, val_auc_3 = eval_vid(vid, val_preds_3.detach().cpu().numpy(), args.data_name)
    val_acc, val_f1, val_auc = eval_vid(vid, val_preds.detach().cpu().numpy(), args.data_name)

    print("test_acc {:.5f}, ALL_test_f1: {:.5f}, ALL_test_auc:{:.5f}".format(val_acc_1, val_f1_1, val_auc_1))
    print("test_acc {:.5f}, ALL_test_f1: {:.5f}, ALL_test_auc:{:.5f}".format(val_acc_2, val_f1_2, val_auc_2))
    print("test_acc {:.5f}, ALL_test_f1: {:.5f}, ALL_test_auc:{:.5f}".format(val_acc_3, val_f1_3, val_auc_3))


    print("ALL_test_acc {:.5f}, ALL_test_f1: {:.5f}, ALL_test_auc:{:.5f}".format(val_acc, val_f1, val_auc))


    results = "best results are acc {:.5f}, f1: {:.5f}, auc:{:.5f} ".format(val_acc, val_f1, val_auc)
    val_results = classification_report(val_lables.cpu().numpy(), val_preds.cpu().numpy(),
                                        target_names=["truth", "deception"])
    print(results)
    print("[INFO] Classification Report:")
    print(val_results)
    print("[INFO] Test Completed.")


if __name__ == "__main__":
    opts = parse_options()

    print('Config:')
    print(f'  - Frames: {opts.len}')
    print(f'  - Train flag: {opts.train_flag}')
    print(f'  - Image size: {opts.size}')
    print(f'  - Depth: {opts.depth}')
    print(f'  - Dataset: {opts.data_name}')

    # log/model save path
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_name = os.path.join(opts.log_dir, f'{now}_{opts.data_name}_{opts.train_flag}_{opts.m_version}_{opts.len}_{opts.blocks}.txt')
    filename = f'{opts.data_name}_{opts.train_flag}_{opts.m_version}_{opts.len}_{opts.blocks}'

    if opts.test:
        only_test(log_name, filename, args=opts)
    else:
        train_test(log_name, filename, args=opts)
