import argparse
import csv
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models


def get_args():
    parser = argparse.ArgumentParser(description="FruitCNN Gradio App")
    parser.add_argument("--model_path",  type=str, default="./outputs/resnet50/best_model_resnet50.pth")
    parser.add_argument("--results_dir", type=str, default="./outputs")
    parser.add_argument("--save_csv",    type=str, default="results.csv")
    parser.add_argument("--port",        type=int, default=7860)
    return parser.parse_args()


class FruitResNet50(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def load_class_names(results_dir):
    class_file = Path(results_dir) / "class_names.txt"
    if not class_file.exists():
        raise FileNotFoundError(
            f"class_names.txt not found in {results_dir}\n"
            "train3_resnet50.py를 먼저 실행해주세요."
        )
    with open(class_file) as f:
        return [line.strip() for line in f if line.strip()]


def find_best_model_info(results_dir, save_csv):
    csv_path = Path(results_dir) / save_csv
    if not csv_path.is_file():
        return None, 0.0
    best_row, best_f1 = None, -1.0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model", "") == "ResNet50_pretrained":
                f1 = float(row["val_f1"])
                if f1 > best_f1:
                    best_f1  = f1
                    best_row = row
    return best_row, best_f1


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@torch.no_grad()
def predict(image, model, class_names, device, top_k=5):
    if image is None:
        return "이미지를 업로드해주세요."
    img_tensor = eval_transform(image).unsqueeze(0).to(device)
    logits     = model(img_tensor)
    probs      = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, top_k)
    lines = []
    for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
        lines.append(f"{i+1}. {class_names[idx]}: {prob*100:.2f}%")
    return "\n".join(lines)


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(args.results_dir)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("train3_resnet50.py --save_model 을 먼저 실행해주세요.")
        return

    model = FruitResNet50(num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {args.model_path}")

    best_row, best_f1 = find_best_model_info(args.results_dir, args.save_csv)
    if best_row:
        description = (
            f"**Model:** ResNet50 (pretrained fine-tuning) | "
            f"**Val F1:** {best_f1:.4f} | "
            f"**Params:** {int(best_row.get('num_params', 0)):,}"
        )
    else:
        description = "**Model:** ResNet50 (pretrained fine-tuning)"

    def predict_fn(image):
        return predict(image, model, class_names, device, top_k=5)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil", label="과일 이미지 업로드"),
        outputs=gr.Textbox(label="예측 결과 (Top-5)", lines=6),
        title="🍎 FruitCNN Classifier — ResNet50",
        description=description,
        flagging_mode="never",
    )

    print(f"\n========================================")
    print(f"브라우저에서 아래 주소로 접속하세요:")
    print(f"http://localhost:{args.port}")
    print(f"========================================\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        max_file_size="10mb",
        quiet=True,
    )

if __name__ == "__main__":
    main()
