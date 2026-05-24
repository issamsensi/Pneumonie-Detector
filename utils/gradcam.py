from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class GradCamResult:
    heatmap: np.ndarray
    overlay_bgr: np.ndarray


class YOLOGradCAM:
    def __init__(self, torch_model, target_layer=None):
        self.torch_model = torch_model
        self.target_layer = target_layer or self._resolve_target_layer(torch_model)
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None

    @staticmethod
    def _resolve_target_layer(torch_model):
        model = getattr(torch_model, "model", None)
        if model is None:
            raise RuntimeError("Ultralytics model does not expose a backbone for Grad-CAM.")

        if hasattr(model, "model"):
            layers = getattr(model, "model")
            if isinstance(layers, (list, tuple)):
                return layers[-2]
            try:
                return layers[-2]
            except Exception:
                pass

        try:
            return model[-2]
        except Exception as exc:
            raise RuntimeError("Unable to resolve Grad-CAM target layer from the YOLO model.") from exc

    def __enter__(self):
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()

    def _save_activations(self, module, inputs, output):  # noqa: D401
        self.activations = output

    def _save_gradients(self, module, grad_inputs, grad_output):  # noqa: D401
        self.gradients = grad_output[0]

    def build_heatmap(self, image_tensor: torch.Tensor) -> np.ndarray:
        self.torch_model.model.zero_grad(set_to_none=True)
        self._detach_inference_buffers()
        with torch.inference_mode(False):
            output = self.torch_model.model(image_tensor)
        score = self._extract_score(output)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        if activations.dim() == 3:
            activations = activations.unsqueeze(0)
        if gradients.dim() == 3:
            gradients = gradients.unsqueeze(0)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value
        return cam

    def _detach_inference_buffers(self) -> None:
        for module in self.torch_model.model.modules():
            for attribute_name in ("strides", "anchors"):
                if not hasattr(module, attribute_name):
                    continue
                attribute_value = getattr(module, attribute_name)
                if torch.is_tensor(attribute_value):
                    setattr(module, attribute_name, attribute_value.clone().detach())
                elif isinstance(attribute_value, (list, tuple)):
                    cloned_values = [value.clone().detach() if torch.is_tensor(value) else value for value in attribute_value]
                    setattr(module, attribute_name, type(attribute_value)(cloned_values))

    @staticmethod
    def _extract_score(output) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError("Unexpected model output for Grad-CAM.")

        if output.ndim == 3:
            if output.shape[1] >= 5:
                return output[:, 4:, :].amax()
            return output.amax()
        if output.ndim == 2:
            return output.amax()
        return output.reshape(-1).amax()


def make_blue_tinted_overlay(original_rgb: np.ndarray, heatmap: np.ndarray, boxes: list[dict]) -> np.ndarray:
    height, width = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized * 255.0, 0, 255))
    jet_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    blue_bg_rgb = np.dstack(
        [
            np.clip(gray * 0.4, 0, 255),
            np.clip(gray * 0.5, 0, 255),
            np.clip(gray * 0.9, 0, 255),
        ]
    ).astype(np.uint8)
    blue_bg_bgr = cv2.cvtColor(blue_bg_rgb, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(blue_bg_bgr, 0.35, jet_heatmap_bgr, 0.65, 0)

    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
        confidence = box["confidence"]
        label = f"{confidence * 100:.1f}%"

        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y1 - text_height - baseline - 8)
        label_y2 = max(0, y1)
        label_x2 = min(width - 1, x1 + text_width + 12)
        cv2.rectangle(overlay_bgr, (x1, label_y1), (label_x2, label_y2), (15, 18, 30), -1)
        cv2.putText(
            overlay_bgr,
            label,
            (x1 + 6, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay_bgr
