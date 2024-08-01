import pandas as pd
import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
import csv
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

line = '\n-----------------------------------------------------------------------'


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Calculations for Conv 1 Layer
        # Input Size = W1*H1*D1 = 24*32*1
        ## Requires 4 hyperparameters:
        ### out_channels: k = 32 **normally
        ### Spatial extend of each one (kernel size), F = 4
        ### Slide size (stride), S = 1
        ### Padding, P = 2
        # Output Size = W2*H2*D2
        ## W2 = ( ( W1 - F + 2(P) ) / S ) + 1 = ( ( 24 - 4 + 2(2) ) / 1 ) + 1 = 25
        ## H2 = ( ( H1 - F + 2(P) ) / S ) + 1 = ( ( 32 - 4 + 2(2) ) / 1 ) + 1 = 33
        ## D2 = k = 32
        ## Output of Conv1 = W2*H2*D2 = 25*33*32

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=2)
        self.batch1 = torch.nn.BatchNorm2d(32)
        self.drop1 = torch.nn.Dropout(0.1)
        self.relu1 = torch.nn.ReLU()

        # Calculations for Pool 1 Layer
        # Input Size = W2*H2*D2 = 25*33*32
        ## Requires 2 hyperparameters:
        ### Spatial extend of each one (kernel size), F = 3
        ### Slide size (stride), S = 2
        # Output Size = W3*H3*D2
        ## W3 = ( ( W2 - F ) / S ) + 1 = ( ( 25 - 3 ) / 2 ) + 1 = 12
        ## H3 = ( ( H2 - F ) / S ) + 1 = ( ( 33 - 3 ) / 2 ) + 1 = 16
        ## Output of Pool1 = W3*H3*D2 = 12*16*32
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)  # default stride is equivalent to the kernel_size

        # Calculations for Conv 2 Layer
        # Input Size = W3*H3*D2 = 12*16*32
        ## Requires 4 hyperparameters:
        ### out_channels: k = 32 **normally
        ### Spatial extend of each one (kernel size), F = 3
        ### Slide size (stride), S = 1
        ### Padding, P = 2
        # Output Size = W4*H4*D3
        ## W4 = ( ( W3 - F + 2(P) ) / S ) + 1 = ( ( 12 - 3 + 2(2) ) / 1 ) + 1 = 14
        ## H4 = ( ( H3 - F + 2(P) ) / S ) + 1 = ( ( 16 - 3 + 2(2) ) / 1 ) + 1 = 18
        ## D3 = 32
        ## Output of Conv2 = W4*H4*D3 = 14*18*32
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.drop2 = torch.nn.Dropout(0.2)
        self.relu2 = torch.nn.ReLU()

        # Calculations for Pool 2 Layer
        # Input Size = W4*H4*D3 = 14*18*32
        ## Requires 2 hyperparameters:
        ### Spatial extend of each one (kernel size), F = 2
        ### Slide size (stride), S = 2
        # Output Size = W5*H5*D3
        ## W5 = ( ( W4 - F ) / S ) + 1 = ( ( 14 - 2 ) / 2 ) + 1 = 7
        ## H5 = ( ( H4 - F ) / S ) + 1 = ( ( 18 - 2 ) / 2 ) + 1 = 9
        ## Output of Pool2 = W5*H5*D3 = 7*9*32
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        #------------------------------------------
        # Calculations for Conv 3 Layer
        # Input Size = W5*H5*D3 = 7*9*32
        ## Requires 4 hyperparameters:
        ### out_channels: k = 16 **normally
        ### Spatial extend of each one (kernel size), F = 2
        ### Slide size (stride), S = 1
        ### Padding, P = 1
        # Output Size = W6*H6*D4
        ## W6 = ( ( W5 - F + 2(P) ) / S ) + 1 = ( ( 7 - 2 + 2(1) ) / 1 ) + 1 = 8
        ## H6 = ( ( H5 - F + 2(P) ) / S ) + 1 = ( ( 9 - 2 + 2(1) ) / 1 ) + 1 = 10
        ## D4 = 16
        ## Output of Conv3 = W6*H6*D4 = 8*10*16
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.batch3 = torch.nn.BatchNorm2d(16)
        self.drop3 = torch.nn.Dropout(0.3)
        self.relu3 = torch.nn.ReLU()

        # Calculations for Pool 3 Layer
        # Input Size = W6*H6*D4 = 8*10*16
        ## Requires 2 hyperparameters:
        ### Spatial extend of each one (kernel size), F = 1
        ### Slide size (stride), S = 1
        # Output Size = W7*H7*D4
        ## W7 = ( ( W6 - F ) / S ) + 1 = ( ( 8 - 1 ) / 1 ) + 1 = 8
        ## H7 = ( ( H6 - F ) / S ) + 1 = ( ( 10 - 1 ) / 1 ) + 1 = 10
        ## Output of Pool3 = W7*H7*D4 = 8*10*16
        self.pool3 = torch.nn.MaxPool2d(kernel_size=1, stride=1)
        # ------------------------------------------


        # Defining the Linear layer
        # Input Size = W7*H7*D4 = 8*10*16
        # Output Size = Number of classes = 4
        self.fc = torch.nn.Linear(16 * 8 * 10, 4)

    # defining the network flow
    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.drop1(out)
        out = self.relu1(out)

        # Max Pool 1
        out = self.pool1(out)

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.drop2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)

        # Conv 3
        out = self.conv3(out)
        out = self.batch3(out)
        out = self.drop3(out)
        out = self.relu3(out)

        # Max Pool 3
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fc(out)

        return out

# list for the training set
x_train, y_train, train_times = [], [], []
x_test, y_test, test_times = [], [], []

lines = '\n-----------------------------------------------------------------------\n'

# setting dataset for test && train
testing_files = ['test-merge']
training_files = ['train-merge']
print(lines)

# reading the csv files testings set
print('Loading data...')
for i in testing_files:
    df = pd.read_csv(str(i) + '.csv', encoding="utf8", on_bad_lines='warn')
    for row in range(len(df)):
        y_test.append(df.iloc[row, -1])
        x_test.append(df.iloc[row, 1:-1])
        # test_times.append(str(df.iloc[row, 0]).replace('/', '-').replace(':', '-').replace('.', '-'))

# reading the csv files trainings set
for j in training_files:
    df = pd.read_csv(str(j) + '.csv', encoding="utf8", on_bad_lines='warn')
    for row in range(len(df)):
        y_train.append(df.iloc[row, -1])
        x_train.append(df.iloc[row, 1:-1])
        # train_times.append(str(df.iloc[row, 0]).replace('/', '-').replace(':', '-').replace('.', '-'))

print('Raw Training Images size: ', len(x_train))
print('Raw Training Label size: ', len(y_train), ' | Unique Label: ', np.unique(np.array(y_train)))
print('Raw Testing Images size: ', len(x_test))
print('Raw Testing Label size: ', len(y_test), ' | Unique Label: ', np.unique(np.array(y_test)))

print(lines)
print('Data preprocessing...')

# แปลงรูปของดาต้าให้อยู่ในรูปแบบของ tensor
x_trainImages = []
x_testImages = []
y_trainLabels = []
y_testLabels = []

# # Plot Heatmap and save to jpg
height = 24
width = 32
#
# for ind in range(len(x_train)):
#     print(ind, y_train[ind])
#     frame2D = []
#     frame = x_train[ind]
#     for h in range(height):
#         frame2D.append([])
#         for w in range(width):
#             t = frame[h * width + w]
#             frame2D[h].append(t)
#
#     p1 = sns.heatmap(frame2D, vmin=20, vmax=34)
#     p1.set_title('train: ' + str(train_times[ind]) + "-" + str(y_train[ind]))
#     plt.show()
#
#     p1.figure.savefig("../Tan-Bank/img/train/" + str(y_train[ind]) + "/" + str(train_times[ind]) + "-" + str(ind) + ".png")
#
# for ind in range(len(x_test)):
#     print(ind, y_test[ind])
#     frame2D = []
#     frame = x_test[ind]
#     for h in range(height):
#         frame2D.append([])
#         for w in range(width):
#             t = frame[h * width + w]
#             frame2D[h].append(t)
#
#     p1 = sns.heatmap(frame2D, vmin=20, vmax=34)
#     p1.set_title('test: ' + str(test_times[ind]) + "-" + str(y_test[ind]))
#     plt.show()
#
#     p1.figure.savefig("../Tan-Bank/img/test/" + str(y_test[ind]) + "/" + str(test_times[ind]) + "-" + str(ind) + ".png")

def transformData(data):
    transformedData = []

    for ind in range(len(data)):
        frame2D = []
        frame = x_test[ind]
        for h in range(height):
            frame2D.append([])
            for w in range(width):
                t = frame[h * width + w]
                frame2D[h].append(t)

        transformedData.append([frame2D])

    return transformedData


x_train_transformData = transformData(x_train)
x_test_transformData = transformData(x_test)

x_trainImages = torch.FloatTensor(x_train_transformData)
y_trainLabels = torch.LongTensor(y_train)
x_testImages = torch.FloatTensor(x_test_transformData)
y_testLabels = torch.LongTensor(y_test)

# Data Loader
print('Transformed X_trainImages Images size: ', x_trainImages.size())
print('Transformed Y_trainLabels Labels size: ', y_trainLabels.size())
print('Transformed X_testImages Images size: ', x_testImages.size())
print('Transformed Y_testLabels Labels size: ', y_testLabels.size())

# ปรับ device ให้เป็น GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(lines)
print('Using {} device'.format(device))
model = CNN().to(device)

# Hyperparameters for training the model (you can change it)

num_epochs = 200
learning_rate = 0.001
weight_decay = 0.001
batch_size = 4 # 8 16 32 64 128 256 512
criterion = torch.nn.CrossEntropyLoss()

# # Model Setting
# model = CNN().to(device)
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# get the training set
train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)  # สร้าง datasets สำหรับ train
train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ train set

# get the test set
test_set = data_utils.TensorDataset(x_testImages, y_testLabels)  # สร้าง datasets สำหรับ test
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)  # สร้าง dataloader สำหรับ test set

# training_set = ConcatDataset([train_loader.dataset])
# testing_set = ConcatDataset([test_loader.dataset])

# Test loss and accuracy
train_score = []
test_score = []

# start training
print(lines)
print('Start training...')
round = [i for i in range(1, 31)]

for i in round:
    print(lines)
    # Model Setting
    model = CNN().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):

        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):  # แบ่งข้อมูลจาก train loader ที่

            data, target = data.to(device), target.to(device)  # ส่งข้อมูลไปยัง GPU
            optimizer.zero_grad()
            model.train()  # สั่งให้ model ทำงานในโหมด train
            outputs = model(data)  # รับค่า output จาก model
            loss = criterion(outputs, target)  # คำนวณค่า loss
            loss.backward()  # คำนวณ gradient
            optimizer.step()  # ปรับปรุงค่า weight
            train_loss += loss.item()  # บวกค่า loss ที่ได้ไปเรื่อยๆ
            _, predicted = torch.max(outputs.data, 1)  # คำนวณค่า accuracy
            train_total = target.size(0)  # นับจำนวนข้อมูลทั้งหมดใน batch
            train_correct = (predicted == target).sum().item()  # นับจำนวนข้อมูลที่ถูกต้องใน batch

        test_acc = 0.0
        test_total = 0.0
        model.eval()  # เป็นการบอก model ว่าเราจะทำการ test แล้ว
        with torch.no_grad():
            for batch_idx, (inputs, y_true) in enumerate(test_loader):
                # Extracting images and target labels for the batch being iterated
                inputs, y_true = inputs.to(device), y_true.to(device)  # ส่งข้อมูลไปยัง GPU

                # Calculating the model output and the cross entropy loss
                outputs = model(inputs)  # รับโมเดลที่เราสร้างไว้

                # Calculating the accuracy of the model
                _, predicted = torch.max(outputs.data, 1)  # predic ด้วยโมเดล จาก data ที่ได้มาจาก testing set
                test_acc += (predicted == y_true).sum().item()  # นับ accuracy
                test_total += y_true.size(0)  # นับจำนวนข้อมูลทั้งหมด

        # คำนวณค่า accuracy และ loss ของ train set และ test set
        if epoch % 2 == 0:
            TrainingLoss = train_loss / len(train_loader)
            Training_Accuracy = train_correct / train_total
            Testing_Accuracy = test_acc / test_total
            print(
                'Round {} \tEpoch: {}\{} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.10f} \tTesting Accuracy: {:.10f}'.format(
                    i, epoch + 2, num_epochs, TrainingLoss, Training_Accuracy, Testing_Accuracy))

    train_score.append(Training_Accuracy)  # สำหรับเก็บค่า accuracy ของ train
    test_score.append(Testing_Accuracy)  # สำหรับเก็บค่า accuracy ของ test

# พล็อตค่าความแม่นยำของแต่ละรอบ
plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.axhline(0.95, color="gold", linestyle="--", alpha=0.5, linewidth=1.0, label='base line')
plt.title("cnn")
plt.plot(round, train_score, '--', label='Train', color="darkgreen", alpha=0.5, linewidth=1.0)
plt.plot(round, test_score, '--', label='Test', color="maroon", alpha=0.5, linewidth=1.0)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Rounds', fontsize=10)
plt.legend(fontsize=12, loc='lower right')
plt.show()