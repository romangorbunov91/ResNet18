import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
#from models.temporal import BasicBlock, ResNet18
#from pathlib import Path



#import torch.nn as nn

from torch.utils.data import DataLoader

# Import Datasets
from datasets.TinyImageNetDataset import TinyImageNetDataset

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class ResNet18Trainer(object):

    def __init__(self, configer):
        self.configer = configer

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory
        '''
        # Losses
        self.losses = {
            'train': AverageMeter(),                      #: Train loss avg meter
            'val': AverageMeter(),                        #: Val loss avg meter
            'test': AverageMeter()                        #: Test loss avg meter
        }

        # Train val and test accuracy
        self.accuracy = {
            'train': AverageMeter(),                      #: Train accuracy avg meter
            'val': AverageMeter(),                        #: Val accuracy avg meter
            'test': AverageMeter()                        #: Test accuracy avg meter
        }
        '''
        # DataLoaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        #self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None
        self.lr = None

        # Training procedure
        self.optimizer = None
        self.iters = None
        self.epoch = 0
        self.train_transforms = None
        self.val_transforms = None
        self.loss = None
        
        self.dataset = self.configer.get("dataset").lower()         #: str: Type of dataset
        '''
        # Tensorboard and Metrics
        self.tbx_summary = SummaryWriter(str(Path(configer.get('checkpoints', 'tb_path'))  #: Summary Writer plot
                                             / configer.get("dataset")                     #: data with TensorboardX
                                             / configer.get('checkpoints', 'save_name')))
        self.tbx_summary.add_text('parameters', str(self.configer).replace("\n", "\n\n"))
        self.save_iters = self.configer.get('checkpoints', 'save_iters')    #: int: Saving ratio
        '''


    def init_model(self):
        """Initialize model and other data for procedure"""
        '''
        self.loss = nn.CrossEntropyLoss().to(self.device)

        # Selecting correct model and normalization variable based on type variable
        self.net = GestureTransoformer(self.backbone, self.in_planes, self.n_classes,
                                       pretrained=self.configer.get("network", "pretrained"),
                                       n_head=self.configer.get("network", "n_head"),
                                       dropout_backbone=self.configer.get("network", "dropout2d"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module")
                                       )

        # Initializing training
        self.iters = 0
        self.epoch = None
        phase = self.configer.get('phase')

        # Starting or resuming procedure
        if phase == 'train':
            self.net, self.iters, self.epoch, optim_dict = self.model_utility.load_net(self.net)
        else:
            raise ValueError('Phase: {} is not valid.'.format(phase))

        if self.epoch is None:
            self.epoch = 0

        # ToDo Restore optimizer and scheduler from checkpoint
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net, self.iters)
        self.scheduler = MultiStepLR(self.optimizer, self.configer["solver", "decay_steps"], gamma=0.1)

        #  Resuming training, restoring optimizer value
        if optim_dict is not None:
            print("Resuming training from epoch {}.".format(self.epoch))
            self.optimizer.load_state_dict(optim_dict)
        '''
        # Selecting Dataset and DataLoader
        if self.dataset == "tinyimagenetdataset":
            Dataset = TinyImageNetDataset
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.train_transforms = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])

            self.val_transforms = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                normalize,
            ])
            
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.train_loader = DataLoader(
            Dataset(self.data_path, split="train", transform=self.train_transforms),
            batch_size=self.configer.get('data', 'batch_size'),
            shuffle=True,
            num_workers=self.configer.get('solver', 'workers'))
        
        self.val_loader = DataLoader(
            Dataset(self.data_path, split="val", transform=self.val_transforms),
            batch_size=self.configer.get('data', 'batch_size'),
            shuffle=False,
            num_workers=self.configer.get('solver', 'workers'))

    def __train(self):
        """Train function for every epoch."""
        
        print(f"Train size: {len(self.train_loader.dataset)}")
        print(f"Val size: {len(self.val_loader.dataset)}")
        print(f"Классов: {len(self.train_loader.dataset.class_names)}")
        
        '''
        
        
        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):
            """
            input, gt
            """
            inputs, gt = data_tuple[0].to(self.device), data_tuple[1].to(self.device)

            output = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss(output, gt.squeeze(dim=1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            predicted = torch.argmax(output.detach(), dim=1)
            correct = gt.detach().squeeze(dim=1)

            self.iters += 1
            self.update_metrics("train", loss.item(), inputs.size(0),
                                float((predicted==correct).sum()) / len(correct))
            '''
    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            # for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=str(self.accuracy["val"].avg))):
            for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=""+str(np.random.randint(200)))):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)
                gt = data_tuple[1].to(self.device)

                output = self.net(inputs)
                loss = self.loss(output, gt.squeeze(dim=1))

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                self.iters += 1
                self.update_metrics("val", loss.item(), inputs.size(0),
                                    float((predicted == correct).sum()) / len(correct))

        self.tbx_summary.add_scalar('val_loss', self.losses["val"].avg, self.epoch + 1)
        self.tbx_summary.add_scalar('val_accuracy', self.accuracy["val"].avg, self.epoch + 1)
        print("VAL  accuracy: {:.4f}".format(self.accuracy["val"].avg))
        accuracy = self.accuracy["val"].avg
        self.accuracy["val"].reset()
        self.losses["val"].reset()

        ret = self.model_utility.save(accuracy, self.net, self.optimizer, self.iters, self.epoch + 1)
        if ret < 0:
            return -1
        elif ret > 0 and self.test_loader is not None:
            self.__test()
        return ret

    def train(self):
        self.__train()
        '''
        for n in range(self.configer.get("epochs")):
            print("Starting epoch {}".format(self.epoch + 1))
            self.__train()
            ret = self.__val()
            if ret < 0:
                print("Got no improvement for {} epochs, current epoch is {}."
                      .format(self.configer.get("checkpoints", "early_stop"), n))
                break
            self.epoch += 1
        '''
    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()