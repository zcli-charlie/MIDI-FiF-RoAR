import tqdm
import torch
import numpy as np
import torch.nn as nn

def training(model, device, train_loader, optimizer):
    model.train()
    
    train_loss, total_acc = 0, 0
    pbar = tqdm.tqdm(train_loader, disable = False)

    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device).long(), y.to(device).long()
        optimizer.zero_grad()
        
        y_hat = model(x)    # (batch, seq, class_probability) = (16,512,4)
        
        output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)       
        output = torch.from_numpy(output).to(device).long()     # (batch, seq)

        attn = (y != 0).float()      # != bar pad word; shape: (16,512)
        acc = torch.sum((output == y).float() * attn)
        acc /= torch.sum(attn)
        total_acc += acc

        loss_func = nn.CrossEntropyLoss(reduction='none')   # (batch, class, ...)
        y_hat = y_hat.permute(0,2,1)
        loss = loss_func(y_hat, y)
        loss = loss * attn
        loss = torch.sum(loss) / torch.sum(attn)
        # update
        loss.backward()

        train_loss += loss
        optimizer.step()

    return train_loss/len(train_loader), total_acc/len(train_loader)


def valid(model, device, valid_loader):
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device).long(), y.to(device).long()
            
            y_hat = model(x) 
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(device).long()

            attn = (y != 0).float()      # != bar pad word; shape: (16,512)

            acc = torch.sum((output == y).float() * attn)
            acc /= torch.sum(attn)
            valid_acc += acc

            loss_func = nn.CrossEntropyLoss(reduction='none')
            y_hat = y_hat.permute(0,2,1)
            loss = loss_func(y_hat, y)
            loss = loss * attn
            loss = torch.sum(loss) / torch.sum(attn)

            valid_loss += loss

    return valid_loss/len(valid_loader), valid_acc/len(valid_loader)
