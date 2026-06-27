import argparse
import os
from collections import OrderedDict
import time
import modeling_finetune #Important
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd

from timm.data import create_transform
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from timm.utils import accuracy
from torchvision import datasets, transforms

import utils
from dataset_folder import ImageFolder
from MALIMG import MalimgDataset as MalimgIM
import sys
from sklearn.metrics import precision_score

trainacclist=[]
testacclist=[]
def evaluate(data_loader, model, device, experiment):
    criterion = torch.nn.CrossEntropyLoss()
    class_to_idx = data_loader.dataset.class_to_idx
    class_map = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_map)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 初始化每个类别的正确预测数量和总预测数量
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # switch to evaluation mode
    model.eval()
    counter = 0
    with open(f'predic_probabilities_{experiment}_test.csv', 'wb') as file:
        file.write('image_name,actual_label,predicted_label_index\n'.encode())
        for batch in data_loader:
            images = batch[0]
            target = batch[-1]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images, target)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            batch_size = images.shape[0]
            _, predicted = torch.max(output, 1)
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

            for i in range(len(output)):
                counter += 1
                index = counter
                tensval = output[i]
                class_assign = torch.argmax(tensval)
                file.write((str(index) + ',' + str(class_assign.item()) + ',' + str(target[i].item()) + '\n').encode())
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # 打印每个类别的精度
    print("Class-wise accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            cls_acc = 100 * class_correct[i] / class_total[i]
            print(f"Class {class_map[i]}: {cls_acc:.2f}%")
        else:
            print(f"Class {class_map[i]}: No samples")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test(data_loader, model, criterion, device, epoch=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if epoch is not None:
        header = f'Epoch: [{epoch}] Test:'

    class_to_idx = data_loader.dataset.class_to_idx
    class_map = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_map)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    all_targets = []
    all_preds = []

    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images, target)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        _, predicted = torch.max(output, 1)

        all_targets.extend(target.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        for i in range(len(target)):
            label = target[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'* {header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    print(f"{header} Class-wise accuracy and precision:")

    precisions = precision_score(all_targets, all_preds, labels=list(range(num_classes)), average=None, zero_division=0)

    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            prec = 100 * precisions[i]
            print(f"Class {class_map[i]}: Accuracy: {acc:.2f}% | Precision: {prec:.2f}%")
        else:
            print(f"Class {class_map[i]}: No samples")
    testacclist.append({'epoch': epoch, 'acc1': metric_logger.acc1.global_avg, 'acc5': metric_logger.acc5.global_avg, 'loss': metric_logger.loss.global_avg})


def train(data_loader, model, criterion, optimizer, device, epoch):
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    class_to_idx = data_loader.dataset.class_to_idx
    class_map = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_map)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images, target)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        batch_size = images.shape[0]
        _, predicted = torch.max(output, 1)
        for i in range(len(target)):
            label = target[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'* Epoch {epoch} Train Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    print(f"Epoch {epoch} Class-wise accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            cls_acc = 100 * class_correct[i] / class_total[i]
            print(f"Class {class_map[i]}: {cls_acc:.2f}%")
        else:
            print(f"Class {class_map[i]}: No samples")    
    trainacclist.append({'epoch': epoch, 'acc1': metric_logger.acc1.global_avg, 'acc5': metric_logger.acc5.global_avg, 'loss': metric_logger.loss.global_avg})

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'MALIMG':
        dataset = MalimgIM(args.data_path,train=is_train,transform=transform,evalt=args.is_eval)
        nb_classes = 25
    elif args.data_set == "image_folder":
        print('loading imagefolder dataset')
        root = os.path.join(args.data_path, '' if is_train else '')

        print('loading ' + root)

        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
    else:
        raise NotImplementedError()
    print('Built dataset')
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def storeout():
    # 使用字典创建 DataFrame
    dataframe = pd.DataFrame(trainacclist) 
    dataframe2 = pd.DataFrame(testacclist) 
    # 将 DataFrame 保存为 CSV 文件
    dataframe.to_csv('./ACC_result/trainacc_output.csv', index=False) 
    dataframe2.to_csv('./ACC_result/testacc_output.csv', index=False)

if __name__ == '__main__':
    if not os.path.exists('training_log.txt'):
        with open('training_log.txt', 'w') as f:
            pass
    original_stdout = sys.stdout
    sys.stdout = open('training_log.txt', 'a')
    parser = argparse.ArgumentParser('MAE fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', #vit_base_patch16_224
                        help='Name of model to train')
    # * Finetuning params
    parser.add_argument('--model_path', default='./outputs/binary.pth', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_mean_pooling', default=True,action='store_true')
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--data_set', default='MALIMG', choices=['CIFAR', 'IMNET', 'MALIMG','image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--data_path',
                        default='./Datasets/malimg_dataset',
                        type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=25, type=int,
                        help='number of the classification types')
    parser.add_argument('--experiment', default='family', type=str,choices=['binary', 'type', 'family'])
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    #freeze parameter
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Training parameters
    parser.add_argument('--is_train',default=True,type=bool,help='train the model')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--step_size', default=15, type=int, help='step size for learning rate scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for learning rate scheduler')
    # Evaluation parameters
    parser.add_argument('--is_eval',default=False,type=bool,help='use eval')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--init_scale', default=0.001, type=float)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.is_train:
        dataset_train, _ = build_dataset(is_train=True, args=args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
        dataset_test,_ = build_dataset(is_train=False, args=args)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    )

    if args.model_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.model_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')

    print("Load ckpt from %s" % args.model_path)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        elif key.startswith('norm.'):
            new_dict[key[3:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    
    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    if args.freeze:
        for n, p in model.named_parameters():
            # Freeze if in args.freeze
            if any(n.startswith(prefix) for prefix in args.freeze):
                p.requires_grad = False
    
    start_time = time.time()
    if args.is_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        for epoch in range(args.epochs):
            train(data_loader_train, model, criterion, optimizer, device, epoch)
            test(data_loader_test, model, criterion, device, epoch)
            scheduler.step()
        checkpoint_path = 'checkpoint_fine.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"模型已保存到 {checkpoint_path}")
        storeout()
    else:
        evaluate(data_loader_val, model, device, args.experiment)
    
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    
    sys.stdout.close()
    sys.stdout = original_stdout
    sys.exit(0)