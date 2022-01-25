import pickle
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import f1_score

class SimpleNN(nn.Module):
    def __init__(self, w_in, w_out, num_classes):
        super(SimpleNN, self).__init__()
        self.hidden_layer = nn.Linear(w_in, w_out)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(w_out, num_classes)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, node_features, X, Y):
        x = F.relu(self.hidden_layer(node_features))
        x = self.dropout(x)
        y_pred = self.output_layer(x)[X]
        loss = self.loss(y_pred, Y)
        return loss, y_pred

with open('data/'+'ACM'+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
with open('data/'+'ACM'+ '/labels.pkl', 'rb') as f:
    labels = pickle.load(f)


node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
train_x = torch.from_numpy(labels[0][:, 0]).type(torch.LongTensor)
train_y = torch.from_numpy(labels[0][:, 1]).type(torch.LongTensor)
val_x = torch.from_numpy(labels[1][:, 0]).type(torch.LongTensor)
val_y = torch.from_numpy(labels[1][:, 1]).type(torch.LongTensor)
test_x = torch.from_numpy(labels[2][:, 0]).type(torch.LongTensor)
test_y = torch.from_numpy(labels[2][:, 1]).type(torch.LongTensor)

num_classes = torch.max(train_y).item() + 1

model = SimpleNN(node_features.shape[1], 128, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

epochs = 40
final_f1 = 0


best_val_loss = 10000
best_test_loss = 10000
best_train_loss = 10000
best_train_f1 = 0
best_val_f1 = 0
best_test_f1 = 0

for i in range(epochs):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.005:
            param_group['lr'] = param_group['lr'] * 0.9
    print('Epoch:  ',i+1)
    model.zero_grad()
    model.train()
    loss,y_train = model(node_features, train_x, train_y)
    train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_y, num_classes=num_classes)).cpu().numpy()
    print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
    loss.backward()
    optimizer.step()
    model.eval()
    # Valid
    with torch.no_grad():
        val_loss, y_valid = model.forward(node_features, val_x, val_y)
        val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), val_y, num_classes=num_classes)).cpu().numpy()
        print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
        test_loss, y_test = model.forward(node_features, test_x, test_y)
        test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_y, num_classes=num_classes)).cpu().numpy()
        print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
    if val_f1 > best_val_f1:
        best_val_loss = val_loss.detach().cpu().numpy()
        best_test_loss = test_loss.detach().cpu().numpy()
        best_train_loss = loss.detach().cpu().numpy()
        best_train_f1 = train_f1
        best_val_f1 = val_f1
        best_test_f1 = test_f1 
print('---------------Best Results--------------------')
print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
final_f1 += best_test_f1