"""
Shows a small example of how to load a pretrain model (VGG16) from PyTorch,
and modifies this to train on the CIFAR10 dataset. The same method generalizes
well to other datasets, but the modifications to the network may need to be changed.

Video explanation: https://youtu.be/U4bHxEhMGNk
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-08 Initial coding

"""

# Imports
import json
import os

import numpy as np
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.optim import Adam, Adagrad

from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
from shutil import rmtree
# # Hyperparameters

#
# learning_rate = 0.001
# # 更改learning_rate:先设置0.001，再设置衰减速率
#
# batch_size = 32
# num_epochs = 5
# optimizer
# save_path


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)



def train_with_resenet50(learning_rate, batch_size, num_epochs, optimizer_name, save_path
                         #, loss_img, accuracy_img
                         ,folder_name
                         ):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 新建文件夹
    cwd = os.getcwd()
    mk_file(os.path.join(cwd, folder_name))


    # data transformation
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "cell_data")  # fruit data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 设置fruit类型
    cell_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cell_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # Hyperparameters

    num_classes = 2

    # learning_rate = 0.001
    # # 更改learning_rate:先设置0.001，再设置衰减速率
    #
    # batch_size = 32
    # num_epochs = 5

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # Simple Identity class that let's input pass without changes
    # class Identity(nn.Module):
    #     def __init__(self):
    #         super(Identity, self).__init__()
    #
    #     def forward(self, x):
    #         return x

    # Load pretrain model & modify it
    model = torchvision.models.resnet50(pretrained=True)

    channel_in = model.fc.in_features

    # If you want to do finetuning then set requires_grad = False
    # Remove these two lines if you want to train entire model,
    # and only want to load the pretrain weights.
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=channel_in, out_features=2, bias=True)

    # model.fc = nn.Sequential(nn.Linear(channel_in, 256),
    #                          nn.ReLU(),
    #                          nn.Dropout(0.4),  # ???????
    #                          nn.Linear(256, num_classes)
    #                          )
    model.to(device)



    # Loss and optimizer

    loss_function = nn.CrossEntropyLoss()  # 网络输出不经 softmax 层，直接由 CrossEntropyLoss 计算交叉熵损失
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)
    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(params, lr=learning_rate)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(params, lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    else:
        print("this optimizer is not supported")




    # 可视化Loss和Accuracy
    train_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    val_loss_list = []

    best_acc_val = 0.0
    best_acc_train = 0.0
    best_loss_val = 1.0
    best_loss_train = 1.0

    train_steps = len(train_loader)
    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        acc_train = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            predict = torch.max(logits, dim=1)[1]
            acc_train += torch.eq(predict, labels.to(device)).sum().item()

            loss.backward()
            optimizer.step()



            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epochs,
                                                                     loss)

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        total_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                val_loss_each_it = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                total_val_loss += val_loss_each_it.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           num_epochs)
        train_accurate = acc_train / train_num
        val_accurate = acc / val_num
        train_loss = running_loss / train_steps # loss/iteration in one epoch
        val_loss = total_val_loss / len(validate_loader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accurate)
        val_accuracy_list.append(val_accurate)
        print('[epoch %d] train_loss: %.3f train_accuracy: %.3f val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps,train_accurate,val_loss, val_accurate))

        if val_accurate > best_acc_val:
            best_acc_val = val_accurate
            torch.save(model.state_dict(), save_path)

        if train_accurate > best_acc_train:
            best_acc_train = train_accurate

        if val_loss < best_loss_val:
            best_loss_val = val_loss

        if train_loss < best_loss_train:
            best_loss_train = train_loss


    a = open(str(folder_name)+'/best.txt', 'w')
    a.write("best_acc_val: " + str(best_acc_val))
    a.write("best_acc_train: " + str(best_acc_train))
    a.write("best_loss_train: " + str(best_loss_train))
    a.write("best_loss_val: " + str(best_loss_val))
    a.close()

    print('Finished Training')
    print('best_acc_val:',best_acc_val)
    print('best_acc_train:', best_acc_train)
    print('best_loss_val:',best_loss_val)
    print('best_loss_train:', best_loss_train)




    print()
    draw_loss_and_accuracy(train_loss_list, val_loss_list, train_accuracy_list,
                           val_accuracy_list, num_epochs
                           #, loss_img, accuracy_img
                           ,folder_name
                           )






def draw_loss_and_accuracy(train_loss_list, val_loss_list, train_accuracy_list,
                           val_accuracy_list, num_epochs
                           #, loss_img, accuracy_img
                           ,folder_name
                           ):

    epochrange = range(1, num_epochs+1)

    # 设置图片的大小
    matplotlib.rc('figure', figsize=(14, 7))  # 单位为厘米
    # 设置字体的大小
    matplotlib.rc('font', size=14)  # size为字体的大小
    # 是否显示背景网格
    matplotlib.rc('axes', grid=True)
    # grid：取为Flase为不显示背景网格，True为显示
    # 背景颜色
    matplotlib.rc('axes', facecolor='black')
    # 白色：white
    # 绿色：green
    # 黄色：yellow
    # 黑色：black
    # 灰色：grey


    plt.plot(epochrange, train_loss_list, linestyle='-.',marker='o',color='lightgreen',linewidth=4)
    plt.plot(epochrange, val_loss_list, linestyle='-.',marker='o',color='tomato',linewidth=4)
    l = plt.legend(['Train', 'Validation'], loc='lower right', frameon=False)
    for text in l.get_texts():
        text.set_color('white')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.savefig(loss_img)
    plt.savefig(str(folder_name)+'/model_loss.jpg')
    plt.close()



    plt.plot(epochrange, train_accuracy_list, linestyle='-.',marker='o',color='lightgreen',linewidth=4)
    plt.plot(epochrange, val_accuracy_list, linestyle='-.',marker='o',color='tomato',linewidth=4)
    l = plt.legend(['Train', 'Validation'], loc='lower right', frameon=False)
    for text in l.get_texts():
        text.set_color('white')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(str(folder_name)+'/model_accuracy.jpg')
    plt.close()








if __name__ == '__main__':
    train_with_resenet50(learning_rate=0.001, batch_size=32, num_epochs=3,
                         optimizer_name="Adagrad", save_path="./resNet50_1.pth"
                         #, loss_img="./resNet50_1/model_loss.jpg"
                         #,accuracy_img="./resNet50_1/model_accuracy.jpg"
                         ,folder_name="resNet50_1"
                         )

