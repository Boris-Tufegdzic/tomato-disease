from PIL import Image
from PIL import ImageFile
from pathlib import Path
import torchvision
from torch import nn
import torch
from torchvision import transforms
import os
from utils import setup_data
from utils import training_utils

if __name__ == '__main__':
    #directories definition
    base_dir = 'C:\\Users\\boris\\OneDrive\\Bureau\\python_projects\\tomato-disease'
    train_dir = os.path.join(base_dir, 'model', 'data', 'train')
    test_dir = os.path.join(base_dir, 'model', 'data', 'valid')

    #device definition
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #We get the pretrained weights from efficientnet_b3 model
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model = torchvision.models.efficientnet_b3(weights=weights).to(device)
    auto_transforms = weights.transforms

    # manual transforms pipeline definition
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
        transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                            std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    ])

    #Optimizer, loss function and number of epochs definitions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    train_dataloader, test_dataloader, class_names, idx_to_class = setup_data.create_dataloaders(train_dir, test_dir, manual_transforms, BATCH_SIZE, NUM_WORKERS)
    num_classes = len(class_names)
    print(idx_to_class)
    #fine-tuning the efficient net b3 for our classification task (we place the right number of out_features)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1536, 
                        out_features=num_classes, # same number of output units as our number of classes
                        bias=True)).to(device)

    #Some images of our dataset are truncated so we execute the line below in order to accept them

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    #Now we can launch the actual training

    training_utils.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS, device)

    #Saving the trained model

    model_save_path = os.path.join(base_dir, "model", "models", "efficient_net.pth")

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)