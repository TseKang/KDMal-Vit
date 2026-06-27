import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from timm.models import create_model
from regenerate_experiment_results import build_dataset, train, test
from regenerate_experiment_results import evaluate as ev
from utils import setup_for_distributed, init_distributed_mode, MetricLogger
import argparse
from timm.utils import accuracy

# 解析参数
parser = argparse.ArgumentParser('Knowledge Distillation Training')
parser.add_argument('--data_path', default='./Datasets/malimg_dataset', type=str, help='dataset path')
parser.add_argument('--data_set', default='MALIMG', type=str, choices=['CIFAR', 'IMNET', 'MALIMG', 'image_folder'],
                    help='Image Net dataset path')
parser.add_argument('--nb_classes', default=25, type=int, help='number of classes')
parser.add_argument('--input_size', default=224, type=int, help='images input size')

# Evaluation parameters
parser.add_argument('--is_eval',default=False,type=bool,help='use eval')
parser.add_argument('--crop_pct', type=float, default=None)
parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
parser.add_argument('--init_scale', default=0.001, type=float)


parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic)')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
# 新增蒸馏参数
parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft target')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss')

args = parser.parse_args()

if not os.path.exists('distillation_log.txt'):
        with open('distillation_log.txt', 'w') as f:
            pass
original_stdout = sys.stdout
sys.stdout = open('distillation_log.txt', 'a')
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建train数据集
train_dataset, _ = build_dataset(is_train=True, args=args)
sampler_train = torch.utils.data.RandomSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=sampler_train, batch_size=32, num_workers=1, pin_memory=True)
#test
test_dataset, _ = build_dataset(is_train=False, args=args)
sampler_test = torch.utils.data.SequentialSampler(test_dataset)
test_loader = torch.utils.data.DataLoader(
    test_dataset, sampler=sampler_test,batch_size=32, num_workers=1, pin_memory=True)
# 加载教师模型，使用 timm 创建 vit_base_patch16_224
teacher_model = create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    ).to(device)
teacher_model.load_state_dict(torch.load('./checkpoint_prompt_10epoch.pth'))
teacher_model.eval()

# 定义学生模型
student_model = create_model(
    'deit_tiny_patch16_224',
    pretrained=False, 
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    attn_drop_rate=args.attn_drop_rate,
    drop_block_rate=None,
    ).to(device)

# 定义损失函数和优化器
criterion_hard = nn.CrossEntropyLoss()
criterion_soft = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(student_model.parameters(), lr=0.01)

# 学生模型测试函数
def test_student_model(data_loader, model, criterion, device, epoch):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Test Student Model Epoch {epoch}:'
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)
            target = batch[-1].to(device)
            
            # 只传入 images，符合模型 forward 方法的参数要求
            output = model(images)
            
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # 打印测试结果
    print(f'* Test Student Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    return metric_logger.acc1.global_avg

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    student_model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    for batch in train_loader:
        images = batch[0].to(device)
        target = batch[-1].to(device)

        # 教师模型前向传播
        with torch.no_grad():
            teacher_output = teacher_model(images)

        # 学生模型前向传播
        student_output = student_model(images)

        # 计算硬标签损失（学生模型预测与真实标签的损失）
        loss_hard = criterion_hard(student_output, target)
        
        # 计算软标签蒸馏损失（学生模型输出与教师模型输出的分布差异）
        soft_teacher_output = nn.functional.softmax(teacher_output / args.temperature, dim=1)
        soft_student_output = nn.functional.log_softmax(student_output / args.temperature, dim=1)
        loss_soft = criterion_soft(soft_student_output, soft_teacher_output) * (args.temperature ** 2)
        
        # 综合损失
        loss = args.alpha * loss_soft + (1 - args.alpha) * loss_hard
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(student_output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # 打印训练统计信息
    metric_logger.synchronize_between_processes()
    print(f'* Epoch {epoch} Train Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')

    # 测试学生模型
    test_student_model(test_loader, student_model, criterion_hard, device, epoch)
    #测试教师模型
    ev(test_loader, student_model, device, 'family')
sys.stdout.close()
sys.stdout = original_stdout
sys.exit(0)