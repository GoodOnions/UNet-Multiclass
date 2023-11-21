import torch
from model import UNET
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms


if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print('Running on the MPS')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

ROOT_DIR = '../datasets/superMario'
IMG_HEIGHT = 240
IMG_WIDTH = 272

MODEL_PATH = "../models/e40_b32_unet.pth"

EVAL = True
PLOT_LOSS = True

def save_predictions(data, model):    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):

            X, y, s = batch # here 's' is the name of the file stored in the root directory
            X, y = X.to(DEVICE), y.to(DEVICE)
            prediction = model(X)

            prediction = torch.argmax(prediction, dim=1).squeeze()
            prediction = prediction.float().detach().cpu().numpy()



def evaluate(path):
    T = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST)
    ])

    val_set = get_supermario_data(
        root_dir=ROOT_DIR,
        split='val',
        transforms=T,
        shuffle=True,
        eval=True
    )
 
    print('Data has been loaded!')

    net = UNET(in_channels=3, classes=6).to(DEVICE)
    map_location = torch.device(DEVICE)
    checkpoint = torch.load(path,map_location=map_location)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{path} has been loaded and initialized')
    save_predictions(val_set, net)

def plot_losses(path):
    checkpoint = torch.load(path,map_location=torch.device(DEVICE))
    losses_train = checkpoint['loss_values']
    loss_on_val = checkpoint['loss_on_val']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch))

    plt.plot(epoch_list, losses_train, label='Training loss')
    plt.plot(epoch_list, loss_on_val, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epoch/s")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if EVAL:
        evaluate(MODEL_PATH)
    if PLOT_LOSS:
        plot_losses(MODEL_PATH)