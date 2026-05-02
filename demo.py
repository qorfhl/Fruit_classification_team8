import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path

# =========================================================
# 1. 설정
# =========================================================
MODEL_PATH = "./outputs/resnet50/best_model_resnet50.pth"
CLASS_NAMES_PATH = "./outputs/class_names.txt"
NUM_CLASSES = 100

# =========================================================
# 2. 모델 로드
# =========================================================
def load_model(model_path, num_classes):
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    state_dict = torch.load(model_path, map_location="cpu")
    backbone.load_state_dict(state_dict)
    backbone.eval()
    return backbone

# =========================================================
# 3. 클래스 이름 로드
# =========================================================
def load_class_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

# =========================================================
# 4. 전처리
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================================
# 5. 예측 함수
# =========================================================
def predict(image):
    # 모델과 클래스 이름 로드 (최초 1회)
    global model, class_names
    if model is None:
        model = load_model(MODEL_PATH, NUM_CLASSES)
        class_names = load_class_names(CLASS_NAMES_PATH)

    # 전처리
    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 추론
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Top-3 결과
    top3_probs, top3_indices = torch.topk(probs, 3)
    result = {}
    for prob, idx in zip(top3_probs.tolist(), top3_indices.tolist()):
        label = class_names[idx]
        result[label] = round(prob, 4)

    return result

# =========================================================
# 6. 전역 변수 초기화
# =========================================================
model = None
class_names = None

# =========================================================
# 7. Gradio 인터페이스
# =========================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="과일 이미지 업로드 (JPG / PNG)"),
    outputs=gr.Label(num_top_classes=3, label="예측 결과 (Top-3)"),
    title="🍎 Fruit Classifier — Team 8",
    description=(
        "ResNet50 (pretrained on ImageNet, fine-tuned on Fruits-100) 모델을 사용합니다.\n"
        "과일 이미지를 업로드하면 Top-3 예측 결과와 신뢰도를 반환합니다.\n"
        "지원 클래스: 100종 (Fruits-100 데이터셋 기준)"
    ),
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # 모델 사전 로드
    print("Loading model...")
    model = load_model(MODEL_PATH, NUM_CLASSES)
    class_names = load_class_names(CLASS_NAMES_PATH)
    print(f"Model loaded. Classes: {len(class_names)}")

    demo.launch(share=False)
