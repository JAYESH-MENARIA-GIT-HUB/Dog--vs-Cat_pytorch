import torch 
import time
import torchvision
import torch.nn.functional as F
from torch import nn


class NN(nn.Module):
  def __init__(self,input_size, output_size,hidden_layers,p_drop=0.5):
    super().__init__()
    self.hidden_layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
    layer_size = zip(hidden_layers[:-1],hidden_layers[1:])
    self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_size])
    self.output = nn.Linear(hidden_layers[-1], output_size)
    self.dropout = nn.Dropout(p=p_drop)

  def forward(self, x):
    for linear in self.hidden_layers:
      x = F.relu(linear(x))
      x = self.dropout(x)
    x = self.output(x)

    return F.log_softmax(x,dim=1) 

def validation(model, testloader,criterion):
  test_loss = 0
  accuracy = 0
  for images, labels in testloader:
    images = images.resize_(images.shape[0],784)

    output = model.forward(images)
    test_loss += criterion(output, labels).item()

    ps = torch.exp(output)

    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
  return test_loss, accuracy

def train(model,trainloader,testloader,criterion,optimizer,epochs = 2,print_every = 40):
  
  steps = 0
  running_loss = 0
  for e in range(epochs):
    model.train()
    for images, labels in trainloader:
      steps += 1

      images.resize_(images.shape[0],784)
      optimizer.zero_grad()
      output = model.forward(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if steps % print_every == 0:
        model.eval()
        with torch.no_grad():
          test_loss, accuracy = validation(model,testloader,criterion)
        print("Epoch:{}/{}....".format(e+1,epochs),
              "Traning_loss:{:.3f}....".format(running_loss/print_every),
              "Test_loss:{:.3f}.....".format(test_loss/len(testloader)),
              "Accuracy:{:.3f}.....".format(accuracy/len(testloader)))
        running_loss = 0
        model.train()
