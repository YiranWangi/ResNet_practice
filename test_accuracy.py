import os
import json
from os import listdir
from os.path import join

import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


import torch.nn as nn
from shutil import rmtree

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = '../../../cell1/pytorch_classification 2/Test5_resnet50/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # load image
    batch_size = 32
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "cell_data")
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))



    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for testing.".format(test_num))

    model = torchvision.models.resnet50()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 2, bias=True)
    model.to(device)
    # load model weights
    weights_path = "./resNet50_unchangedAdagrad0.01.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch

    with torch.no_grad():
        val_bar = tqdm(test_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    
    test_accurate = acc / test_num

    cwd = os.getcwd()
    mk_file(os.path.join(cwd, "best"))

    a = open( 'best/best.txt', 'w')
    a.write("test_acc: " + str(test_accurate))
    a.close()
    
    print("test accuracy is {}".format(test_accurate))
    






if __name__ == '__main__':
    main()
