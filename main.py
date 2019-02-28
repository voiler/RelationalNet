import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from dataset import SortOfClevr
from models import RelationalNet


def train(epoch):
    model.train()
    loader = tqdm(train_loader)
    for batch in loader:
        x, r_qst, r_a, n_qst, n_a = map(lambda x: x.to(args.device), batch)
        optimizer.zero_grad()
        output = model(x, r_qst)
        loss = loss_function(output, r_a)
        loss.backward()
        optimizer.step()
        correct = output.argmax(1).eq(r_a.data).sum().item()
        r_accuracy = correct * 100. / len(r_a)
        optimizer.zero_grad()
        output = model(x, n_qst)
        loss = loss_function(output, n_a)
        loss.backward()
        optimizer.step()
        correct = output.argmax(1).eq(n_a.data).sum().item()
        n_accuracy = correct * 100. / len(n_a)
        loader.set_description(
            'Epoch:{} Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                epoch, r_accuracy, n_accuracy))


def test(epoch):
    model.eval()
    r_accuracies = []
    n_accuracies = []
    loader = tqdm(test_loader)
    for batch in loader:
        x, r_qst, r_a, n_qst, n_a = map(lambda x: x.to(args.device), batch)
        output = model(x, r_qst)
        correct = output.argmax(1).eq(r_a.data).sum().item()
        r_accuracy = correct * 100. / r_a.shape[0]
        r_accuracies.append(r_accuracy)
        output = model(x, n_qst)
        correct = output.argmax(1).eq(n_a.data).sum().item()
        n_accuracy = correct * 100. / n_a.shape[0]
        n_accuracies.append(n_accuracy)
        loader.set_description(
            'Epoch:{} Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                epoch, r_accuracy, n_accuracy))
    r_accuracy = sum(r_accuracies) / len(r_accuracies)
    n_accuracy = sum(n_accuracies) / len(n_accuracies)
    print('Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%'.format(
        r_accuracy, n_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    torch.manual_seed(args.seed)
    train_data = SortOfClevr('./data')
    test_data = SortOfClevr('./data', False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    model = RelationalNet().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        torch.save(model.state_dict(), './model/model.pth')
