import torch
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.metrics import classification_report

# 1. Initialize environment & data pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_TEST_DIR = "/content/drive/MyDrive/DIP/Local_Model_Data/test"

eval_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

local_test_dataset = datasets.ImageFolder(LOCAL_TEST_DIR, eval_transforms)
local_test_loader = torch.utils.data.DataLoader(local_test_dataset, batch_size=16, shuffle=False)

# 2. Re-create and load the distilled 18-class Student architecture
local_student_model = models.resnet18(weights=None)
local_student_model.fc = nn.Linear(local_student_model.fc.in_features, 18)
local_student_model.load_state_dict(torch.load("local_student_periocular.pth", map_location=device))
local_student_model = local_student_model.to(device)

# 3. Generate granular evaluation matrix
def generate_detailed_metrics(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(classification_report(all_labels, all_preds, target_names=dataloader.dataset.classes, digits=4))

generate_detailed_metrics(local_student_model, local_test_loader)