

```python
# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
import torch
```


```python
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
```


```python
# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()
```

    Requirement already satisfied: gputil in /usr/local/lib/python2.7/dist-packages
    Requirement already satisfied: numpy in /usr/local/lib/python2.7/dist-packages (from gputil)
    [33mYou are using pip version 9.0.3, however version 10.0.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Requirement already satisfied: psutil in /usr/local/lib/python2.7/dist-packages
    [33mYou are using pip version 9.0.3, however version 10.0.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Requirement already satisfied: humanize in /usr/local/lib/python2.7/dist-packages
    [33mYou are using pip version 9.0.3, however version 10.0.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    ('Gen RAM Free: 9.4 GB', ' I Proc size: 2.4 GB')
    GPU RAM Free: 923MB | Used: 10516MB | Util  92% | Total 11439MB



```python

#Training VGG-Att3-concat-pc

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

import torch

#######

import torch
import torchvision
import torchvision.transforms as transforms
transform10 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))])
batch_size = 128
lr = 0.1
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform10)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

######

# memory footprint support libraries/code
import psutil
import humanize
import os
import GPUtil as GPU
#GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
#gpu = GPUs[0]
#def printm():
  #process = psutil.Process(os.getpid())
  #print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
  #print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
#printm()

######
print("Training model VGG-Att3-concat-pc on CIFAR 10")
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


class Net(nn.Module):
  def __init__(self):
          super(Net, self).__init__()
           
        
       
       
        
          #FIRST LAYER
          self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
          self.bn1 =  nn.BatchNorm2d(64)
          self.drop1 = nn.Dropout2d(p=0.3)
          #SECOND LAYER
          self.conv2 =nn.Conv2d(64, 64, kernel_size=3, padding=1)
          self.bn2 =  nn.BatchNorm2d(64)
          #THIRD LAYER
          self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
          self.bn3 =  nn.BatchNorm2d(128)
          self.drop3 = nn.Dropout2d(p=0.4)
          #4TH LAYER
          self.conv4 =nn.Conv2d(128, 128, kernel_size=3, padding=1)
          self.bn4 =  nn.BatchNorm2d(128)
          
          #5TH LAYER
          self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
          self.bn5 =  nn.BatchNorm2d(256)
          self.drop5 = nn.Dropout2d(p=0.4)
          
          #6TH LAYER
          self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
          self.bn6 =  nn.BatchNorm2d(256)
          self.drop6 = nn.Dropout2d(p=0.4)
          
          #7TH LAYER
          self.conv7 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
          self.bn7 =  nn.BatchNorm2d(256)
          self.pool7 = nn.MaxPool2d(2, stride=2)
          
         
          #8TH LAYER
          self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
          self.bn8 =  nn.BatchNorm2d(512)
          self.drop8 = nn.Dropout2d(p=0.4)
        
          #9TH LAYER
          self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn9 =  nn.BatchNorm2d(512)
          self.drop9 = nn.Dropout2d(p=0.4)
           
        
          #10TH LAYER
          self.conv10 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn10 =  nn.BatchNorm2d(512)
          self.pool10 = nn.MaxPool2d(2, stride=2)
          
       
          #11TH LAYER
          self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn11 =  nn.BatchNorm2d(512)
          self.drop11 = nn.Dropout2d(p=0.4)
           
          
          #12TH LAYER
          self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn12 =  nn.BatchNorm2d(512)
          self.drop12 = nn.Dropout2d(p=0.4)
           
          
          #13TH LAYER
          self.conv13 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn13 =  nn.BatchNorm2d(512)
          self.pool13 = nn.MaxPool2d(2, stride=2)
         
          #14TH LAYER
          self.conv14 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn14 =  nn.BatchNorm2d(512)
          self.pool14 = nn.MaxPool2d(2, stride=2)
            
        
          #15TH LAYER
          self.conv15 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn15 =  nn.BatchNorm2d(512)
          self.pool15 = nn.MaxPool2d(2, stride=2)
           
          #16TH LAYER
          
          self.drop16 = nn.Dropout2d(p=0.5)
          self.linear16 = nn.Linear(512, 512)
          self.bn16 =  nn.BatchNorm2d(512)
         
        
        
        
          #self.weightUA = MyModuleUA((1,512)).cuda()#vector u for layer 10
          #self.weightUB = MyModuleUB((1,512)).cuda()# vector u for layer 13
          self.layerp = nn.MaxPool1d(kernel_size =  2, dilation = 1, padding = 0, stride=2)
          self.bnF =  nn.BatchNorm2d(512)
          self.fc1 = nn.Linear(512, 512)
          self.linearUA = nn.Linear(512, 1, bias = False)
          self.linearUB = nn.Linear(512, 1, bias = False)
          self.linearUC = nn.Linear(256, 1, bias = False)
          #self.dropF = nn.Dropout2d(p=0.5)
          self.fc3 = nn.Linear(1024 + 256, 10)

            
  def forward(self, x):
          size = x.size(0)
          x = self.drop1(F.relu(self.bn1(self.conv1(x)),inplace = True))#1ST LAYER
          x = (F.relu(self.bn2(self.conv2(x)),inplace = True))#2ND LAYER
          x = self.drop3(F.relu(self.bn3(self.conv3(x)),inplace = True))#3rd LAYER
          x = (F.relu(self.bn4(self.conv4(x)),inplace = True))#4th LAYER
          x = self.drop5(F.relu(self.bn5(self.conv5(x)),inplace = True))#5th LAYER
          x = self.drop6(F.relu(self.bn6(self.conv6(x)),inplace = True))#6th LAYER
          x = self.pool7(F.relu(self.bn7(self.conv7(x)),inplace = True))#7th LAYER
          out7 = x.view(x.size(0), x.size(1),-1).clone() #feature vectors from layer7
          x = self.drop8(F.relu(self.bn8(self.conv8(x)),inplace = True))#8th LAYER
          x = self.drop9(F.relu(self.bn9(self.conv9(x)),inplace = True))#9th LAYER
          x = self.pool10(F.relu(self.bn10(self.conv10(x)),inplace = True))#10th LAYER
         
          out10 = x.view(x.size(0), x.size(1),-1).clone() #feature vectors from layer 10
          x = self.drop11(F.relu(self.bn11(self.conv11(x)),inplace = True))#11th LAYER
          x = self.drop12(F.relu(self.bn12(self.conv12(x)),inplace = True))#12th LAYER
          x = self.pool13(F.relu(self.bn13(self.conv13(x)),inplace = True))#13th LAYER
          out13 = x.view(x.size(0), x.size(1),-1).clone() # feature vectors from layer 13
          x = self.pool14(F.relu(self.bn14(self.conv14(x)),inplace = True))#14th LAYER
          x = self.pool15(F.relu(self.bn15(self.conv15(x)),inplace = True))#15th LAYER
          
          x = x.view(x.size(0), -1) 
         
          x = (F.relu(self.bn16(self.linear16((self.drop16(x) ))), inplace = True))#16th LAYER
         
          x = x.view(-1, 512)
         
          outModified = x.view(x.size(0), x.size(1),1).clone()
          outModified2 = x.view(x.size(0),1, x.size(1)).clone()
          outModified2 = self.layerp(outModified2)
          outModified2 = outModified2.view(outModified2.size(0), outModified2.size(2),1)
         
          out10add = out10.add(outModified) #L(i)+ g layer 10
          out13add = out13.add(outModified)#L(i)+ g layer 13
          out7add = out7.add(outModified2)#L(i)+ g layer 7
          
          
          #layer 13 dot product
          out13add = out13add.view(out13add.size(0), out13add.size(2), out13add.size(1))
          outRes = self.linearUA(out13add)
          outRes = outRes.view(outRes.size(0),-1)
          outRes = F.softmax(outRes, dim = 1)
        
        
          #layer 13 soft max and weighted addition
          outRes2 = outRes.view(outRes.size(0), outRes.size(1), 1) 
         
          outDot2 = torch.bmm(out13, outRes2)#weights*l(i)
         
          outDot2 = outDot2.view(outDot2.size(0), outDot2.size(1))
          
          #layer 10 dot product
        
          out10add = out10add.view(out10add.size(0), out10add.size(2), out10add.size(1))
         
          outRes3 = self.linearUB(out10add)
          outRes3 = outRes3.view(outRes3.size(0),-1)
         
          outRes3 = F.softmax(outRes3, dim = 1)
        
          
          
        
          #layer 10 soft max and weighted addition
          
          outRes4 = outRes3.view(outRes3.size(0), outRes3.size(1), 1)
         
          outDot4 = torch.bmm(out10, outRes4)#weights*l(i)
         
          outDot4 = outDot4.view(outDot4.size(0), outDot4.size(1))
          
           #layer 7 dot product
        
          out7add = out7add.view(out7add.size(0), out7add.size(2), out7add.size(1))
         
          outRes5 = self.linearUC(out7add)
          outRes5 = outRes5.view(outRes5.size(0),-1)
         
          outRes5 = F.softmax(outRes5, dim = 1)
        
         
          
        
          #layer 7 soft max and weighted addition
          
          outRes6 = outRes5.view(outRes5.size(0), outRes5.size(1), 1)
         
          outDot6 = torch.bmm(out7, outRes6)#weights*l(i)
         
          outDot6 = outDot6.view(outDot6.size(0), outDot6.size(1))
        
          #concat layers
         
          outConcat = torch.cat((outDot2, outDot4, outDot6), dim = 1)
          
          outConcat = outConcat.view(outConcat.size(0),-1)
          
        
          x = F.softmax((self.fc3((outConcat)) ), dim =1)
          
          return x



import torch.backends.cudnn as cudnn
net = Net().cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
import torch.optim as optim
import time



criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
start = time.time()
logfile = open('vgg-att3-concat-pc.csv', 'a+')
dataheader = 'epoch, iteration, loss, train-acc, lr, time \n'
logfile.write(dataheader)
logfile.close()
for epoch in range(300):  # loop over the dataset multiple times
    
    if epoch %25 == 0 :
        if epoch != 0:
            lr = lr/2.0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 5e-4)

    running_loss = 0.0
    total = 0
    correct = 0
    torch.save(net.state_dict(), 'checkpointModelAtt3.pkl')
    for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print(len(labels))
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda(async = True)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            total += labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels.data).cpu().sum()
            
            if i % 10 == 9:    # print every 10 mini-batches
                        end = time.time()
                        timetaken = (end - start)
                
                        data = ('[%d, %5d] loss: %.3f, Acc: %.3f%% (%d/%d), Learning-rate:%.8f' %
                        (epoch + 1, i + 1, running_loss / 10,100.*correct/total, correct, total, lr))
                        
                        print(str(data) + ' time: ' + str(timetaken))
                        
                        dataline = ('%d , %5d ,  %.3f ,  %.3f , %.8f , %.8f' %
                        (epoch + 1 +250, i + 1, running_loss / 10,100.*correct/total, lr, timetaken))
                        
                        logfile = open('vgg-att3-concat-pc.csv', 'a+')
                        logfile.write(str(dataline) + '\n')
                        logfile.close()
                        start = time.time()
                        running_loss = 0.0
                        total = 0
                        correct = 0
                        
            
    lr = lr - 1e-7
    if lr <= 0:
	     break;
```


```python

#testing vgg-att3-concat-pc model
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform10)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)
import torch.backends.cudnn as cudnn
net = Net().cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
import torch.optim as optim
import time

dict_loaded = torch.load('checkpointModelAtt3.pkl')
dict_keys_loaded = dict_loaded.keys()
new_keys = net.state_dict().keys()
dictionary = dict(zip(dict_keys_loaded, new_keys))

new_dict = dict((dictionary[key], value) for (key, value) in dict_loaded.items())
print(new_dict.keys())
net.load_state_dict((new_dict ))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cls_pred, cls_true):
        num_classes = 10
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)
        plt.imshow(cm, interpolation='nearest',cmap=plt.cm.jet)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        #plt.titlr('Confusion Matrix')
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.show()
        
        

test_loss = 0
correct = 0
total = 0
net.eval()
preds = []
trues = []
for batch_idx, (inputs, targets) in enumerate(testloader):
        
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda(async = True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        preds +=(Variable(predicted).data).cpu().numpy().tolist()
        trues += targets.data.cpu().numpy().tolist()
        
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        data = ( 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(data)
          
print(preds)
print(trues)

print(data)
plt.figure()
plot_confusion_matrix(preds, trues)
unique, counts = np.unique(trues, return_counts=True)
dict(zip(unique, counts))

```


```python

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

import torch

#######

import torch
import torchvision
import torchvision.transforms as transforms
transform10 = transforms.Compose(
                                 [transforms.ToTensor(),
                                  transforms.Normalize((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))])
batch_size = 128
lr = 0.1
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform10)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

######

# memory footprint support libraries/code
import psutil
import humanize
import os
import GPUtil as GPU
#GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
#gpu = GPUs[0]
#def printm():
#process = psutil.Process(os.getpid())
#print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
#print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
#printm()

######
print("Training model VGG-Att2-concat-pc on CIFAR 10")
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        
        
        
        #FIRST LAYER
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 =  nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(p=0.3)
        #SECOND LAYER
        self.conv2 =nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 =  nn.BatchNorm2d(64)
        #THIRD LAYER
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 =  nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.4)
        #4TH LAYER
        self.conv4 =nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 =  nn.BatchNorm2d(128)
        
        #5TH LAYER
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 =  nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(p=0.4)
        
        #6TH LAYER
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 =  nn.BatchNorm2d(256)
        self.drop6 = nn.Dropout2d(p=0.4)
        
        #7TH LAYER
        self.conv7 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 =  nn.BatchNorm2d(256)
        self.pool7 = nn.MaxPool2d(2, stride=2)
        
        
        #8TH LAYER
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 =  nn.BatchNorm2d(512)
        self.drop8 = nn.Dropout2d(p=0.4)
        
        #9TH LAYER
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 =  nn.BatchNorm2d(512)
        self.drop9 = nn.Dropout2d(p=0.4)
        
        
        #10TH LAYER
        self.conv10 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 =  nn.BatchNorm2d(512)
        self.pool10 = nn.MaxPool2d(2, stride=2)
        
        
        #11TH LAYER
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 =  nn.BatchNorm2d(512)
        self.drop11 = nn.Dropout2d(p=0.4)
        
        
        #12TH LAYER
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 =  nn.BatchNorm2d(512)
        self.drop12 = nn.Dropout2d(p=0.4)
        
        
        #13TH LAYER
        self.conv13 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 =  nn.BatchNorm2d(512)
        self.pool13 = nn.MaxPool2d(2, stride=2)
        
        #14TH LAYER
        self.conv14 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn14 =  nn.BatchNorm2d(512)
        self.pool14 = nn.MaxPool2d(2, stride=2)
        
        
        #15TH LAYER
        self.conv15 =nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn15 =  nn.BatchNorm2d(512)
        self.pool15 = nn.MaxPool2d(2, stride=2)
        
        #16TH LAYER
        
        self.drop16 = nn.Dropout2d(p=0.5)
        self.linear16 = nn.Linear(512, 512)
        self.bn16 =  nn.BatchNorm2d(512)
            
            
            
     
        
        self.bnF =  nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 512)
        self.linearUA = nn.Linear(512, 1, bias = False)
        self.linearUB = nn.Linear(512, 1, bias = False)
        
        self.fc3 = nn.Linear(1024, 10)
                    
                    
    def forward(self, x):
        size = x.size(0)
        x = self.drop1(F.relu(self.bn1(self.conv1(x)), inplace = True))#1ST LAYER
        x = (F.relu(self.bn2(self.conv2(x)), inplace = True))#2ND LAYER
        x = self.drop3(F.relu(self.bn3(self.conv3(x)), inplace = True))#3rd LAYER
        x = (F.relu(self.bn4(self.conv4(x)), inplace = True))#4th LAYER
        x = self.drop5(F.relu(self.bn5(self.conv5(x)), inplace = True))#5th LAYER
        x = self.drop6(F.relu(self.bn6(self.conv6(x)), inplace = True))#6th LAYER
        x = self.pool7(F.relu(self.bn7(self.conv7(x)), inplace = True))#7th LAYER
        x = self.drop8(F.relu(self.bn8(self.conv8(x)), inplace = True))#8th LAYER
        x = self.drop9(F.relu(self.bn9(self.conv9(x)), inplace = True))#9th LAYER
        x = self.pool10(F.relu(self.bn10(self.conv10(x)), inplace = True))#10th LAYER

        out10 = x.view(x.size(0), x.size(1),-1).clone() #feature vectors from layer 10
        x = self.drop11(F.relu(self.bn11(self.conv11(x)), inplace = True))#11th LAYER
        x = self.drop12(F.relu(self.bn12(self.conv12(x)), inplace = True))#12th LAYER
        x = self.pool13(F.relu(self.bn13(self.conv13(x)), inplace = True))#13th LAYER
        out13 = x.view(x.size(0), x.size(1),-1).clone() # feature vectors from layer 13
        x = self.pool14(F.relu(self.bn14(self.conv14(x)), inplace = True))#14th LAYER
        x = self.pool15(F.relu(self.bn15(self.conv15(x)), inplace = True))#15th LAYER

        x = x.view(x.size(0), -1)
        x = (F.relu(self.bn16(self.linear16((self.drop16(x) ))), inplace = True))#16th LAYER
    
        x = x.view(-1, 512)
        x = (F.relu(self.bnF(self.fc1(x)), inplace = True))
        outModified = x.view(x.size(0), x.size(1),1).clone()
        
       
        out10add = out10.add(outModified) #L(i)+ g layer 10
        out13add = out13.add(outModified)#L(i)+ g layer 13

          #layer 13 dot product
        out13add = out13add.view(out13add.size(0), out13add.size(2), out13add.size(1))
        outRes = self.linearUA(out13add)
        outRes = outRes.view(outRes.size(0),-1)
            
        outRes = F.softmax(outRes, dim = 1)
               
                
                 #layer 13 soft max and weighted addition
        outRes2 = outRes.view(outRes.size(0), outRes.size(1), 1)
                
        outDot2 = torch.bmm(out13, outRes2)#weights*l(i)
                    
        outDot2 = outDot2.view(outDot2.size(0), outDot2.size(1))
                    
                    #layer 10 dot product
                    
        out10add = out10add.view(out10add.size(0), out10add.size(2), out10add.size(1))
       
        outRes3 = self.linearUB(out10add)
        outRes3 = outRes3.view(outRes3.size(0),-1)
        
        outRes3 = F.softmax(outRes3, dim = 1)
          
            
            
            #layer 10 soft max and weighted addition
            
        outRes4 = outRes3.view(outRes3.size(0), outRes3.size(1), 1)
            
        outDot4 = torch.bmm(out10, outRes4)#weights*l(i)
                
        outDot4 = outDot4.view(outDot4.size(0), outDot4.size(1))
                
                
                #concat layers
                
        outConcat = torch.cat((outDot2, outDot4), dim = 1)
                    
        outConcat = outConcat.view(outConcat.size(0),-1)
                    
                    
        x = F.softmax((self.fc3((outConcat)) ), dim =1)
                        
        return x

import torch.backends.cudnn as cudnn
net = Net().cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
import torch.optim as optim
import time

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
start = time.time()
logfile = open('vggAtt2concatpc.csv', 'a+')
dataheader = 'epoch, iteration, loss, train-acc, lr, time \n'
logfile.write(dataheader)
logfile.close()
for epoch in range(300):  # loop over the dataset multiple times
    
    if epoch % 25 == 0:
        if epoch != 0:
            lr = lr/2.0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 5e-4)

    running_loss = 0.0
    total = 0
    correct = 0
    torch.save(net.state_dict(), 'checkpointModelAtt2.pkl')
    for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print(len(labels))
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda(async = True)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            total += labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels.data).cpu().sum()
            
            if i % 10 == 9:    # print every 10 mini-batches
                        end = time.time()
                        timetaken = (end - start)
                
                        data = ('[%d, %5d] loss: %.3f, Acc: %.3f%% (%d/%d), Learning-rate:%.8f' %
                        (epoch + 1, i + 1, running_loss / 10,100.*correct/total, correct, total, lr))
                        stri = ('%.3f%%' % (100.*correct/total)) 
                        print(str(data) + ' time: ' + str(timetaken))
                        #dataline = str(epoch + 1) + ', ' + str(i+1) + ', ' + str(running_loss/10) + ', ' + str(timetaken) + ', ' + str(lr)+','+ str(stri)
                        dataline = ('%d , %5d ,  %.3f ,  %.3f , %.8f , %.8f' %
                        (epoch + 1, i + 1, running_loss / 10,100.*correct/total, lr, timetaken))
                        
                        logfile = open('vggAtt2concatpc.csv', 'a+')
                        logfile.write(str(dataline) + '\n')
                        logfile.close()
                        start = time.time()
                        running_loss = 0.0
                        total = 0
                        correct = 0
            
    lr = lr - 1e-7
    if lr <= 0:
	break;
            
print('Finished Training')
```


```python

#testing vgg-att2-concat-pc model
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform10)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)
import torch.backends.cudnn as cudnn
net = Net().cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
import torch.optim as optim
import time

dict_loaded = torch.load('checkpointModelAtt2.pkl')
dict_keys_loaded = dict_loaded.keys()
new_keys = net.state_dict().keys()
dictionary = dict(zip(dict_keys_loaded, new_keys))

new_dict = dict((dictionary[key], value) for (key, value) in dict_loaded.items())
print(new_dict.keys())
net.load_state_dict((new_dict ))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cls_pred, cls_true):
        num_classes = 10
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)
        plt.imshow(cm, interpolation='nearest',cmap=plt.cm.jet)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        #plt.titlr('Confusion Matrix')
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.show()
        

test_loss = 0
correct = 0
total = 0
net.eval()
preds = []
trues = []
for batch_idx, (inputs, targets) in enumerate(testloader):
        
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda(async = True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        preds +=(Variable(predicted).data).cpu().numpy().tolist()
        trues += targets.data.cpu().numpy().tolist()
        
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        data = ( 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(data)
          
print(preds)
print(trues)

print(data)
plt.figure()
plot_confusion_matrix(preds, trues)
unique, counts = np.unique(trues, return_counts=True)
dict(zip(unique, counts))
```
