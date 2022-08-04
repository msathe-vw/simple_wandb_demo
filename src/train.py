from utils.config_utils import combined_parser
from utils.engine import evaluate, train_one_epoch
import torch
import utils.utils as utils
import datetime
import os
import time

from datasets.penn_fudan import PennFudanDataset
from models.mask_rcnn import get_mask_rcnn
import datasets.transforms as T
import wandb
from PIL import Image

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    # W&B Log the training dataset as an artifact here
    wandb.log_artifact(args.train, name='PennFudan', type='pedestrian_dataset')

    # use our dataset and defined transformations
    dataset = PennFudanDataset(args.train, get_transform(train=True))
    dataset_test = PennFudanDataset(args.train, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_mask_rcnn(num_classes)
    # move model to the right device
    model.to(device)
    # W&B Watches model and logs model topology, gradients, and/or params.
    wandb.watch(model, log_freq=50, log_graph=True)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)


    # W&B Resume functionality currently doesn't work
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore(args.checkpoint), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)
        if args.output_dir:
            save_path = os.path.join(args.output_dir, 'model_{}.pth'.format(epoch))
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch},
                save_path)

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)
        # put the model in evaluation mode
        model.eval()
        originals = []
        predictions = []
        with torch.no_grad():
            for i in range(10):
                img, _ = dataset_test[i]
                prediction = model([img.to(device)])
                # W&B can handle Numpy, Torch, or TF tensors as images easily
                originals.append(wandb.Image(img))
                predictions.append(wandb.Image(prediction[0]['masks'][0, 0]))
        wandb.log({"Val Predictions": predictions, "Val Images": originals})
    
    # W&B log last model    
    art = wandb.Artifact(f'maskrcnn--{wandb.run.id}', type="model")
    art.add_file(save_path, save_path.split('/')[-1])

if __name__ == "__main__":
    args = combined_parser()
    # W&B initialise here
    wandb.init(project="ped_detection0", config=dict(args))
    main(args)