import torch
from tqdm import tqdm
from sklearn.metrics import rand_score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def test_loss(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks, weights in tqdm.tqdm(iter(test_loader), "Test"):
            images, masks, weights = images.to(device), masks.to(device), weights.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks, weights)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)



def restore(images, mean=0.5, std=0.225):
    images = images.detach().cpu().numpy() * np.array(std) + np.array(mean)
    images.resize((images.shape[0], *images.shape[-2:]))
    return images

def output_to_mask(images):
    images = torch.argmax(images, dim=1).to(torch.uint8)
    return images.cpu()

def pixel_error(ground_truth, prediction):
  if ground_truth.size() != prediction.size():
    raise ValueError("Input masks must have the same shape.")

  misclassified_pixels = (ground_truth != prediction).sum()
  total_pixels = ground_truth.numel()
  error = misclassified_pixels / total_pixels

  return error

def rand_error(ground_truth, prediction):
  if ground_truth.size() != prediction.size():
    raise ValueError("Input masks must have the same shape.")

  error = 1.0 - rand_score(ground_truth.view(-1).numpy(), prediction.view(-1).numpy())

  return error

def evaluate_error(model, data):
    model.to(device)
    
    instance, masks, _ = data
    instance = instance.to(device)
    
    output = model(instance)
    output = output_to_mask(output)
    
    masks = masks.view(output.size())

    pixel_err, rand_err = 0.0, 0.0

    for i in range(len(instance)):
        pixel_err += pixel_error(masks[i], output[i])
        rand_err  += rand_error(masks[i], output[i])

    return pixel_err / len(instance), rand_err / len(instance)

def plot_output(model, data):
    model.to(device)
    
    instance, masks, _ = data
    instance = instance.to(device)
    
    output = model(instance)
    output = output_to_mask(output)
    
    masks    = masks.view(output.size())
    instance = restore(instance)
    
    for i in range(len(instance)):

        image = cv2.resize(instance[i], masks[i].size(), interpolation=cv2.INTER_CUBIC)
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 5, 1)
        plt.imshow(output[i] * 255, cmap="gray")  
        plt.axis('off')
        plt.title("Pred Mask") 
        
        plt.subplot(1, 5, 2)
        plt.imshow(masks[i] * 255, cmap="gray")  
        plt.axis('off')
        plt.title("True Mask")
    
        plt.subplot(1, 5, 3)
        plt.imshow(image, cmap="gray")  
        plt.axis('off')
        plt.title("Original Image")
    
        plt.subplot(1, 5, 4)
        plt.imshow(image * output[i].numpy(), cmap="gray")  
        plt.axis('off')
        plt.title("Pred Masked Image")
    
        plt.subplot(1, 5, 5)
        plt.imshow(image * masks[i].numpy(), cmap="gray")  
        plt.axis('off')
        plt.title("True Masked Image")

        print(f"Pixel Error: {pixel_error(masks[i], output[i])}")
        print(f"Rand Error : {rand_error(masks[i], output[i])}")
    
        plt.tight_layout()
    
        plt.show()

def visualize_predictions(model, test_loader, num_samples=3):
   
    c = 0
    for data in test_loader:
        if c >= num_samples:
            break
        print("#" * 10)
        plot_output(model, data)  
        c += 1
        
def evaluate_model(model, test_loader):
   
    avg_pixel_error = 0.0
    avg_rand_error  = 0.0
    
    for data in test_loader:
        errors = evaluate_error(model, data)  
        avg_pixel_error += errors[0]
        avg_rand_error  += errors[1]
    
    avg_pixel_error /= len(test_loader)
    avg_rand_error  /= len(test_loader)
    
    print(f"Average Pixel Error: {avg_pixel_error}")
    print(f"Average Rand Error : {avg_rand_error}")
    
    return avg_pixel_error, avg_rand_error
