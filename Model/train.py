import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score
from monai.metrics import compute_dice

batch_size = 1
epochs = 10
learning_rate = 0.0003  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = 'final_model.pth'  

image_dir = 'numpy_image'
label_dir = 'numpy_label'
dataset = MRISegmentationDataset(image_dir, label_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet3D_FPN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = combined_dice_cross_entropy_loss

def train(model, dataloader, optimizer, criterion, epochs, device, model_save_path):
    epoch_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels, num_classes=4)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

       
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

  
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path} after {epochs} epochs.")

   
    plt.plot(range(1, epochs+1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    return epoch_losses


epoch_losses = train(model, dataloader, optimizer, criterion, epochs, device, model_save_path)
print('Model Trained')

