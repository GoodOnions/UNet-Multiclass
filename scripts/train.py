import torch 
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print('Running on the MPS')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

# Dataset settings
ROOT_DIR = '../datasets/superMario'
IMG_HEIGHT = 240
IMG_WIDTH = 272
IN_CHANNELS = 3
CLASSES = 6

# Training settings
BATCH_SIZE = 20
LEARNING_RATE = 0.0001      # Update from 0.0004
EPOCHS = 10

# Fine tuning settings
MODEL_PATH = '../models/dummy_train.pth'
LOAD_MODEL = False

def train_function (data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data): 
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
    
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data.set_postfix(**{'loss (batch)': loss.item()})
        loss_values.append(loss.item())

    avg_loss = sum(loss_values)/len(loss_values)
    return avg_loss

def validation_function (data, model, loss_fn, device):
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)

        loss = loss_fn(preds, y)
        data.set_postfix(**{'loss (val batch)': loss.item()})
        loss_values.append(loss.item())

    avg_loss = sum(loss_values)/len(loss_values)
    return avg_loss
        

def main():
    global epoch
    epoch = 0 # epoch is initially assigned to 0. If LOAD_MODEL is true then
              # epoch is set to the last value + 1. 
    LOSS_VALS = [] # Defining a list to store loss values after every epoch on training data
    LOSS_ON_VAL = [] # Defining a list to store loss values after every epoch on validation data
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    ]) 

    train_set = get_supermario_data(
        split='train',
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
        shuffle=True,           # To reduce overfitting
    )

    val_set = get_supermario_data(
        split='val',
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
    )

    print('Data Loaded Successfully!')

    # Defining the model, optimizer and loss function
    unet = UNET(in_channels=IN_CHANNELS, classes=CLASSES).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=255) 

    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        LOSS_ON_VAL = checkpoint['loss_on_val']
        print("Model successfully loaded!")    

    # Training the model for every epoch.
    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')

        #Training
        loss_val = train_function(train_set, unet, optimizer, loss_function, DEVICE)
        LOSS_VALS.append(loss_val)

        # Validation
        with torch.no_grad():
            val_loss = validation_function(val_set, unet, loss_function, DEVICE)
            LOSS_ON_VAL.append(val_loss)

        print(f'Training avg Loss: {loss_val}')
        print(f'Validation avg Loss: {val_loss}')

        # Saving the model after every epoch
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e+1,
            'loss_values': LOSS_VALS,
            'loss_on_val': LOSS_ON_VAL
        }, MODEL_PATH)
        print("Epoch completed and model successfully saved!")

    print("Training completed!")


if __name__ == '__main__':
    main()