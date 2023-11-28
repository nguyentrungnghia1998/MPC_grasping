import argparse
import datetime
import json
import logging
import os
import sys
import functools
 
import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torch.optim import AdamW
from torchsummary import summary
from tqdm import tqdm
 
from diffusion.resample import create_named_schedule_sampler
from diffusion.fp16_util import MixedPrecisionTrainer
 
from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from utils.model_util import create_diffusion
from mpc_grasp.dataset import MyDataset
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
 
    # Network
    parser.add_argument('--network', type=str, default='mpcgrasp',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
 
    # Datasets
    parser.add_argument('--dataset', type=str, default="grasp-anywhere",
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default = "data/grasp-anything/",
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')
 
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=300,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
 
    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset')
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere/seen',
                        help='Specific for Grasp-Anywhere')
   
    args = parser.parse_args()
    return args
 
 
def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()
 
   
 
    results = {
        'correct': 0,
        'failed': 0,
        'sum': 0,
    }
 
    ld = len(val_data)
 
    with torch.no_grad():
        for img, query, poses in tqdm(val_data):
            img = img.to(device)
 
            alpha = 0.4
            idx = torch.ones(img.shape[0]).to(device)
 
   
            x0 = net.initial_state(img.shape[0],device)
            output = net.get_output(x0, img, query, alpha, idx)
            output = torch.sigmoid(output)
            output = output.detach().cpu()
 
 
            # sample = sample_fn(
            #     net,
            #     pos_gt.shape,
            #     pos_gt,
            #     img,
            #     query,
            #     alpha,
            #     idx,
            # )
 
 
            iou_coef = iou(output, poses)
 
            delta_angle = torch.abs(output[:,-1] - poses[:,-1])*180
 
            iou_coef_mask = iou_coef > 0.25
            delta_angle_mask = delta_angle < 30
 
            mask = iou_coef_mask & delta_angle_mask
 
            results["correct"] += torch.sum(mask).item()
            results["sum"] += mask.shape[0]
 
    results["failed"] = results['sum'] - results["correct"]
 
    return results
 
def iou(predict, target):
    # Calculate IoU
    # Input:
    #   predict: [batch_size, 5]
    #   target: [batch_size, 5]
    # Output:
    #   iou: [batch_size]
 
    # Calculate area of predict and target
    area_predict = predict[:, 2] * predict[:, 3]
    area_target = target[:, 2] * target[:, 3]
 
    # Calculate intersection
    # Calculate left-top and right-bottom of intersection
    intersection_left_top = torch.max(predict[:, :2] - predict[:, 2:4] / 2, target[:, :2] - target[:, 2:4] / 2)
    intersection_right_bottom = torch.min(predict[:, :2] + predict[:, 2:4] / 2, target[:, :2] + target[:, 2:4] / 2)
 
    # Calculate width and height of intersection
    intersection_wh = intersection_right_bottom - intersection_left_top
    intersection_wh = torch.max(intersection_wh, torch.zeros_like(intersection_wh))
 
    # Calculate area of intersection
    area_intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
 
    # Calculate union
    area_union = area_predict + area_target - area_intersection
 
    # Calculate IoU
    iou = area_intersection / area_union
 
    return iou
 
 
def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }
 
    use_fp16 = False
    fp16_scale_growth = 1e-3
 
    net.train()
 
    # Setup for DDPM
 
    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for img, query, poses in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
 
            img = img.to(device)
            poses = poses.to(device)
 
            if epoch>0:
                alpha = 0.4
            else:
                alpha = 0.4*min(1,batch_idx/len(train_data))
            idx = torch.zeros(img.shape[0]).to(device)
            x0 = net.initial_state(img.shape[0],device)
            t = torch.randint(0, 20, size = (img.shape[0], 1), device=device)
           
            x_pred, u_pred = net(x0, t, img, query, alpha, idx)
 
            # Backward loss
            # mp_trainer.backward(loss)
            # mp_trainer.optimize(optimizer)
 
            lossd = net.compute_loss(poses, x_pred, u_pred)
            loss = lossd['loss']
 
            if batch_idx % 50 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.mean().item()))
 
            results['loss'] += loss
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx
 
    return results
 
 
def run():
    args = parse_args()
 
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
 
    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)
 
    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)
 
    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
 
    # Get the compute device
    device = get_device(args.force_cpu)
 
    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    # Dataset = get_dataset(args.dataset)
    Dataset = MyDataset
    dataset = Dataset(args.dataset_path,args.add_file_path)
    logging.info('Dataset size is {}'.format(dataset.length))
 
    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))
 
    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
 
    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')
 
    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(input_channels=input_channels,
                    reff_dim=768,
                    state_dim=5,
                    control_dim=128,
                    time_dim=64,
                    hidden_dim=512,
                    dropout=args.use_dropout,
                    prob=args.dropout_prob,
                    channel_size=args.channel_size,
                    time_step=20)
 
    net = net.to(device)
    logging.info('Done')
 
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
 
    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()
 
    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
 
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
 
        # # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
 
        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        # for n, l in test_results['losses'].items():
        #     tb.add_scalar('val_loss/' + n, l, epoch)
 
        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou
 
 
if __name__ == '__main__':
    run()
