"""End-to-end TF-C + DINOv3 training and export for InGesture."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_files.InGesture_Configs import Config
from dataloader import Load_Dataset
from loss import NTXentLoss_poly
from model import TFC, target_classifier


def contrastive_loss(model, time_view, time_aug, freq_view, freq_aug, batch_size, config, device):
    h_t, z_t, h_f, z_f = model(time_view, freq_view)
    h_t_aug, _, h_f_aug, _ = model(time_aug, freq_aug)
    criterion = NTXentLoss_poly(device, batch_size, config.Context_Cont.temperature, True)
    return 0.2 * (criterion(h_t, h_t_aug) + criterion(h_f, h_f_aug)) + criterion(z_t, z_f)


def run_pretrain(model, loader, optimizer, config, device, epochs, on_epoch=None):
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for step, (time_view, _, time_aug, freq_view, freq_aug) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            tensors = [item.float().to(device) for item in (time_view, time_aug, freq_view, freq_aug)]
            loss = contrastive_loss(model, *tensors, time_view.shape[0], config, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"pretrain {epoch}/{epochs} batch {step}/{len(loader)}", flush=True)
        print(f"pretrain epoch {epoch}/{epochs}: loss={total / len(loader):.5f}", flush=True)
        if on_epoch:
            on_epoch(epoch, {"loss": total / len(loader)})


def run_finetune(model, classifier, loader, model_optimizer, classifier_optimizer, device, epochs, on_epoch=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        total, correct, count = 0.0, 0, 0
        for step, (time_view, labels, _, freq_view, _) in enumerate(loader, 1):
            time_view, freq_view = time_view.float().to(device), freq_view.float().to(device)
            labels = labels.long().to(device)
            model_optimizer.zero_grad(set_to_none=True)
            classifier_optimizer.zero_grad(set_to_none=True)
            _, z_t, _, z_f = model(time_view, freq_view)
            logits = classifier(torch.cat((z_t, z_f), dim=1))
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 5.0)
            model_optimizer.step()
            classifier_optimizer.step()
            total += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
            if step % 50 == 0:
                print(f"finetune {epoch}/{epochs} batch {step}/{len(loader)}", flush=True)
        print(f"finetune epoch {epoch}/{epochs}: loss={total / count:.5f}, accuracy={correct / count:.4f}", flush=True)
        if on_epoch:
            on_epoch(epoch, {"loss": total / count, "accuracy": correct / count})


@torch.no_grad()
def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, count = 0, 0
    confusion = torch.zeros(8, 8, dtype=torch.long)
    for time_view, labels, _, freq_view, _ in loader:
        time_view, freq_view = time_view.float().to(device), freq_view.float().to(device)
        labels = labels.long().to(device)
        _, z_t, _, z_f = model(time_view, freq_view)
        predictions = classifier(torch.cat((z_t, z_f), dim=1)).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        count += len(labels)
        for truth, prediction in zip(labels.cpu(), predictions.cpu()):
            confusion[truth, prediction] += 1
    return correct / count, confusion


def classification_metrics(confusion, class_names):
    matrix = confusion.double()
    true_positive = matrix.diag()
    support = matrix.sum(1)
    precision = true_positive / matrix.sum(0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "test_accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "evaluated_windows": int(matrix.sum()),
        "confusion_matrix": confusion.tolist(),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
    }


def cap_per_class(payload, limit, seed):
    """Use only training/validation labels to select a bounded reproducible subset."""
    if not limit:
        return payload
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for label in range(8):
        indices = (payload["labels"] == label).nonzero().flatten()
        chosen.append(indices[torch.randperm(len(indices), generator=generator)[:limit]])
    indices = torch.cat(chosen)
    return {"samples": payload["samples"][indices], "labels": payload["labels"][indices]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../../datasets/InGestureOffline"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../artifacts/gesture_model"))
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--local-model-dir", type=Path,
                        help="Offline DINOv3 folder containing config.json and pretrained weights")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--unfreeze-dinov3", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use a tiny random backbone only to verify the pipeline")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train two full ViT-S/16 DINOv3 encoders without pretrained weights")
    parser.add_argument("--train-per-class", type=int, default=0, help="0=all training windows")
    parser.add_argument("--validation-per-class", type=int, default=0, help="0=all validation windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if sum((bool(args.local_model_dir), args.smoke_test, args.from_scratch)) > 1:
        parser.error("Choose only one: --local-model-dir, --from-scratch, --smoke-test")
    if args.pretrain_epochs < 0 or args.finetune_epochs < 1:
        parser.error("Need non-negative pretrain epochs and at least one fine-tune epoch")
    if args.train_per_class < 0 or args.validation_per_class < 0:
        parser.error("Subset limits must be non-negative")
    if args.smoke_test and args.output_dir.name == "gesture_model":
        parser.error("Smoke-test outputs must use a separate directory, e.g. artifacts/smoke_test")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "model_bundle.pt").exists():
        parser.error("Output already contains a bundle; choose a new output directory to preserve it")

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run tools/prepare_ingesture_data.py first; missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(metadata["class_names"]) != 8:
        raise ValueError("The prepared dataset must contain exactly 8 classes")
    train_payload = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=False)
    test_payload = torch.load(args.data_dir / "test.pt", map_location="cpu", weights_only=False)
    train_payload = cap_per_class(train_payload, args.train_per_class, args.seed)
    validation_path = args.data_dir / "val.pt"
    val_payload = cap_per_class(torch.load(validation_path, map_location="cpu", weights_only=False),
                                args.validation_per_class, args.seed) if validation_path.exists() else None
    if len(train_payload["labels"]) < 2:
        raise ValueError("Training data must contain at least two windows")

    config = Config()
    config.TSlength_aligned = int(metadata["window_size"])
    config.dinov3_model_name = args.model_name
    config.dinov3_local_model_dir = args.local_model_dir
    config.dinov3_image_size = 64 if args.smoke_test else args.image_size
    config.dinov3_train_from_scratch = args.from_scratch
    config.dinov3_freeze_backbone = not (args.unfreeze_dinov3 or args.from_scratch)
    config.lr = args.learning_rate
    config.dinov3_local_files_only = bool(args.local_model_dir)
    config.dinov3_random_init_for_smoke_test = args.smoke_test
    config.num_classes_target = len(metadata["class_names"])
    batch_size = min(args.batch_size, len(train_payload["labels"]))
    if batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    device = torch.device(args.device)
    print(f"Device: {device}; source: {'random full ViT-S/16' if args.from_scratch else 'tiny test' if args.smoke_test else 'pretrained'}", flush=True)
    print(f"Train={len(train_payload['labels'])}; validation={len(val_payload['labels']) if val_payload else 0}; test={len(test_payload['labels'])}", flush=True)

    pretrain_data = Load_Dataset(train_payload, config, "pre_train", subset=False)
    supervised_data = Load_Dataset(train_payload, config, "fine_tune_test", subset=False)
    test_data = Load_Dataset(test_payload, config, "fine_tune_test", subset=False)
    pretrain_loader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # InGesture is strongly imbalanced because the neutral class is much larger.
    # Inverse-frequency sampling prevents the classifier from learning to always
    # predict the majority class; the test set remains untouched.
    # Load_Dataset shuffles internally, so weights must follow its actual label order.
    label_counts = torch.bincount(supervised_data.y_data.long(), minlength=config.num_classes_target).float()
    if (label_counts == 0).any():
        raise ValueError("Every gesture class must have training examples")
    sample_weights = (1.0 / label_counts)[supervised_data.y_data.long()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # BatchNorm in projectors requires at least two examples per training batch.
    train_loader = DataLoader(supervised_data, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(Load_Dataset(val_payload, config, "fine_tune_test"), batch_size=batch_size,
                            shuffle=False) if val_payload else None

    model = TFC(config).to(device)
    classifier = target_classifier(config).to(device)
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model_optimizer = torch.optim.Adam(model_parameters, lr=config.lr, weight_decay=3e-4)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr, weight_decay=3e-4)
    history = []
    best_score = -1.0
    best_epoch = None
    checkpoint_path = args.output_dir / "best_validation.pt"
    def record_epoch(stage, epoch, values):
        nonlocal best_score, best_epoch
        entry = {"stage": stage, "epoch": epoch, **values}
        if stage == "finetune" and val_loader:
            _, confusion = evaluate(model, classifier, val_loader, device)
            score = classification_metrics(confusion, metadata["class_names"])["macro_f1"]
            entry["validation_macro_f1"] = score
            print(f"validation macro F1={score:.4f}", flush=True)
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save({"model": model.state_dict(), "classifier": classifier.state_dict()}, checkpoint_path)
        history.append(entry)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    run_pretrain(model, pretrain_loader, model_optimizer, config, device, args.pretrain_epochs,
                 on_epoch=lambda epoch, values: record_epoch("pretrain", epoch, values))
    run_finetune(model, classifier, train_loader, model_optimizer, classifier_optimizer, device, args.finetune_epochs,
                 on_epoch=lambda epoch, values: record_epoch("finetune", epoch, values))
    if best_epoch is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        classifier.load_state_dict(checkpoint["classifier"])
    # The held-out test set is evaluated once, after all model selection.
    accuracy, confusion = evaluate(model, classifier, test_loader, device)
    print(f"test accuracy={accuracy:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "dinov3_backbone_config": model.time_encoder.backbone.config.to_dict(),
        "class_names": metadata["class_names"],
        "metadata": metadata,
        "model_options": {"input_channels": 3, "image_size": config.dinov3_image_size, "freeze_backbone": True},
        "metrics": classification_metrics(confusion, metadata["class_names"]),
        "training_info": {
            "initialization": "random_full_vits16" if args.from_scratch else "smoke_test" if args.smoke_test else "pretrained",
            "pretrain_epochs": args.pretrain_epochs, "finetune_epochs": args.finetune_epochs,
            "train_windows_used": len(train_payload["labels"]),
            "validation_windows_used": len(val_payload["labels"]) if val_payload else 0,
            "best_validation_epoch": best_epoch, "seed": args.seed,
            "image_size": config.dinov3_image_size, "batch_size": batch_size,
            "learning_rate": args.learning_rate,
            "dataset": metadata.get("dataset", "InGesture v5"),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        },
    }
    torch.save(bundle, args.output_dir / "model_bundle.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_info.json").write_text(
        json.dumps(bundle["training_info"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported: {args.output_dir / 'model_bundle.pt'}")


if __name__ == "__main__":
    main()
