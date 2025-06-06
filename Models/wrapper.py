from tqdm import tqdm
import torch
from Models.utils import set_seed, CS_Div, CS_QMI_normalized
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE_loss = nn.CrossEntropyLoss()

def train_CSMVIB(trainloader, testloader, fold_idx, net, class_num):
    optimizer, scheduler = get_optimizer_scheduler(net)
    best_acc = 0
    best_result = []

    for epoch in range(500):
        net.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, labels = data['views'], data['label']
            inputs, labels = [input.to(device) for input in inputs], labels.to(device)
            optimizer.zero_grad()
            _, outputs, _, features = net(inputs)
            encoded_labels = one_hot(labels.long(), class_num)
            
            loss = compute_loss(inputs, outputs, features, encoded_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # if args.scheduler:
        #     scheduler.step()

        print(f"Epoch {epoch}, Loss: {running_loss/len(trainloader):.4f}")
        
        report, y_pred, y_true, soft = test_CSMVIB(testloader, net)
        if report[0] > best_acc:
            best_acc = report[0]
            best_result = report

    return best_result, y_pred, y_true, soft

def get_optimizer_scheduler(net):

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
    return optimizer, scheduler

def compute_loss(inputs, outputs, features, encoded_labels):

    I_xz = sum([CS_QMI_normalized(input_i.view(input_i.shape[0], -1), z_i, sigma=1) for (input_i, z_i) in zip(inputs, features)])
    inputs_cat = torch.cat(inputs, dim=1)
    CS_div = CS_Div(inputs_cat.view(inputs_cat.shape[0], -1), encoded_labels, outputs, sigma=1)
    loss = CS_div + 0.0001 * I_xz

    return loss

def test_CSMVIB(testloader, net):
    net.eval()
    y_true, y_pred, softs = [], [], []

    with torch.no_grad():
        for data in tqdm(testloader, total=len(testloader)):
            inputs, labels = data['views'], data['label']
            inputs, labels = [input.to(device) for input in inputs], labels.to(device)
            _, outputs, _, _ = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            soft_prediction = outputs.data

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            softs.extend(soft_prediction)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return [report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']], y_pred, y_true, softs
