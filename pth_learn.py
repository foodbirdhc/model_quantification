import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

NEED_MODEL_ARCHITECTURE=True
if NEED_MODEL_ARCHITECTURE:
    neural_planner_path = os.path.join(project_dir, "neural_planner-shaobing_update_code_gate-PTQ-learn")
    sys.path.insert(0, neural_planner_path)
    from model_architecture.model_main import Model_Float
    from common.config import Config

from align_dumper import AlignDumper

import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.fx as fx
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
from collections import OrderedDict
import json
import importlib

# ── 模型输入规格 ──────────────────────────────────────────────
INPUT_NAMES  = ["obstacles", "traffic_cones", "lines", "stoplines", "crosswalks"]
OUTPUT_NAMES = ["scores", "control_points"]
DUMMY_SHAPES = [
    (1, 64, 21, 15),
    (1,  8,  6),
    (1, 64, 20,  6),
    (1,  8, 20,  3),
    (1,  8, 20,  3),
]

import argparse
import importlib
import json
import torch
import torch.nn as nn

def save_layer_outputs_to_file(layer_outputs, output_dir):
    """
    保存每一层输出，和 ONNX 版本保持一致：
      - npy/xxx.npy
      - json/xxx.json
    """
    out_dir = Path(output_dir)
    (out_dir / "npy").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    def _save_single(name, value):
        # convert torch.Tensor -> numpy, accept numpy already
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            # unsupported scalar or object -> try to convert
            try:
                arr = np.asarray(value)
            except Exception:
                return

        np_path = out_dir / "npy" / f"{name}.npy"
        json_path = out_dir / "json" / f"{name}.json"

        np.save(np_path, arr)

        # compute stats if numeric
        try:
            stats = {
                "layer_name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "nan_count": int(np.isnan(arr).sum()),
                "inf_count": int(np.isinf(arr).sum()),
            }
        except Exception:
            stats = {
                "layer_name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

    for tensor_name, tensor in layer_outputs.items():
        # handle nested containers
        if isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
            _save_single(tensor_name, tensor)
        elif isinstance(tensor, (tuple, list)):
            for i, v in enumerate(tensor):
                _save_single(f"{tensor_name}.{i}", v)
        elif isinstance(tensor, dict):
            for k, v in tensor.items():
                _save_single(f"{tensor_name}.{k}", v)
        else:
            # attempt to convert and save
            _save_single(tensor_name, tensor)

def _to_cpu_value(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, tuple):
        return tuple(_to_cpu_value(v) for v in x)
    if isinstance(x, list):
        return [_to_cpu_value(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_cpu_value(v) for k, v in x.items()}
    return x

class NodeOutputRecorder(fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.node_outputs = OrderedDict()

    def _record(self, name, value):
        value = _to_cpu_value(value)
        if isinstance(value, tuple):
            for i, v in enumerate(value):
                self.node_outputs[f"{name}.{i}"] = v
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.node_outputs[f"{name}.{i}"] = v
        elif isinstance(value, dict):
            for k, v in value.items():
                self.node_outputs[f"{name}.{k}"] = v
        else:
            self.node_outputs[name] = value

    def call_module(self, target, args, kwargs):
        out = super().call_module(target, args, kwargs)
        self._record(str(target), out)
        return out

    def call_function(self, target, args, kwargs):
        out = super().call_function(target, args, kwargs)
        name = getattr(target, "__name__", str(target))
        self._record(name, out)
        return out

    def call_method(self, target, args, kwargs):
        out = super().call_method(target, args, kwargs)
        self._record(str(target), out)
        return out


def collect_node_outputs_by_hook(model, inputs):
    outputs = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                outputs[name] = out.detach().cpu().numpy()
            elif isinstance(out, (tuple, list)):
                outputs[name] = tuple(
                    x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in out
                )
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()

    return outputs


def run_torch_node_export(model, output_dir="./output/layer_outputs/torch"):
    """
    生成和 ONNX node-level 保存粒度一致的 PyTorch 输出。
    """
    np.random.seed(5)
    torch.manual_seed(5)

    input_feed = {}
    for input_name, input_shape in zip(INPUT_NAMES, DUMMY_SHAPES):
        input_feed[input_name] = torch.randn(*input_shape, dtype=torch.float32)

    inputs = (
        input_feed["obstacles"],
        input_feed["traffic_cones"],
        input_feed["lines"],
        input_feed["stoplines"],
        input_feed["crosswalks"],
    )

    node_outputs = collect_node_outputs_by_hook(model, inputs)
    save_layer_outputs_to_file(node_outputs, output_dir)
    return node_outputs

def get_all_output_tensor(model, device="cpu"):
    model.eval()
    model.to(device)

    np.random.seed(5)
    torch.manual_seed(5)

    input_feed = {}
    for input_name, input_shape in zip(INPUT_NAMES, DUMMY_SHAPES):
        tensor = torch.randn(*input_shape, dtype=torch.float32, device=device)
        input_feed[input_name] = tensor

    layer_outputs = OrderedDict()

    def _unwrap_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)):
            for item in x:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(x, dict):
            for v in x.values():
                if isinstance(v, torch.Tensor):
                    return v
        return None

    def make_hook(name):
        def hook(moudle, inputs, outputs):
            tensor = _unwrap_tensor(outputs)
            if tensor is not None:
                layer_outputs[name] = tensor.detach().cpu()
        return hook

    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            outputs = model(
                input_feed["obstacles"],
                input_feed["traffic_cones"],
                input_feed["lines"],
                input_feed["stoplines"],
                input_feed["crosswalks"],
            )
        if isinstance(outputs, (tuple, list)):
            if len(outputs) >= 2:
                layer_outputs["final_scores"] = outputs[0].detach().cpu()
                layer_outputs["final_control_points"] = outputs[1].detach().cpu()
            else:
                layer_outputs["final_output"] = outputs[0].detach().cpu()
        else:
            layer_outputs["final_output"] = outputs.detach().cpu()
    finally:
        for h in handles:
            h.remove()
    return layer_outputs, input_feed

def run_torch_modle(model, device="cpu"):
    model.eval()
    model.to(device)

    np.random.seed(5)
    torch.manual_seed(5)

    input_feed = {}
    for input_name, input_shape in zip(INPUT_NAMES, DUMMY_SHAPES):
        tensor = torch.randn(*input_shape, dtype=torch.float32, device=device)
        input_feed[input_name] = tensor

    try:
        with torch.no_grad():
            outputs = model(
                input_feed["obstacles"],
                input_feed["traffic_cones"],
                input_feed["lines"],
                input_feed["stoplines"],
                input_feed["crosswalks"],
            )
    except Exception:
        print(f"Run torch model failed")
    return input_feed

def load_torch_model(model_path, model_class=None, device="cpu", model_kwargs=None, align_dumper_obj=None):
    checkpoint = torch.load(model_path, map_location=device)

    # 1) 直接是一个 nn.Module
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device).eval()

    # 2) dict 包了一层 model
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), nn.Module):
            return checkpoint["model"].to(device).eval()

        # 3) dict 里是 state_dict
        if "state_dict" in checkpoint:
            if model_class is None:
                raise ValueError("This checkpoint contains state_dict; you must pass --model-class, e.g. my_package.model:MyModel")
            module_name, class_name = model_class.split(":", 1)
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)

            model = model_cls(**(model_kwargs or {})).to(device).eval()
            model.load_state_dict(checkpoint["state_dict"])
            return model

        # 4) 直接就是 state_dict
        if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            """加载 Model_Float，返回 (model, config)。"""
            pth_path = Path(model_path)
            config_path = pth_path.parent / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"缺少 config.json，期望路径：{config_path}\n"
                    "请将 config.json 放在与 .pth 相同的目录下。"
                )
            config = Config.load(str(config_path))
            model = Model_Float(config, mode="control_point_float", align_dumper_obj=align_dumper_obj)
        
            try:
                ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(pth_path, map_location="cpu")
        
            new_state = OrderedDict()
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                # 历史 key 兼容映射
                if k == "initial_decoder.mode_emb.weight":
                    k = "initial_decoder.mode_emb.emb.weight"
                new_state[k] = v
        
            missing, unexpected = [], []
            try:
                model.load_state_dict(new_state, strict=True)
            except RuntimeError:
                model.load_state_dict(new_state, strict=False)
                model_keys = set(model.state_dict().keys())
                missing     = sorted(model_keys - set(new_state.keys()))
                unexpected  = sorted(set(new_state.keys()) - model_keys)
        
            if missing:
                print(f"[load_model] 缺少 key（{len(missing)} 个），前5个：{missing[:5]}")
            if unexpected:
                print(f"[load_model] 多余 key（{len(unexpected)} 个），前5个：{unexpected[:5]}")
        
            model.eval()
            return model

    raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

def main():
    ap = argparse.ArgumentParser(description="PyTorch Model Layer Output Extraction")
    ap.add_argument("-m", "--model", required=True, help="Path to the .pth checkpoint file")
    ap.add_argument("-c", "--model_class", required=False, help="Model class in the format 'module_name:ClassName'")
    ap.add_argument("--model_kwargs", required=False, help="JSON string of additional keyword arguments for the model class constructor")
    ap.add_argument("--device", default="cpu", help="Device to run the model on (default: cpu)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    align_dumper=AlignDumper(
        output_dir="./output/layer_outputs/torch"
    )
    model = load_torch_model(
        args.model,
        model_class=args.model_class,
        device=device,
        model_kwargs=model_kwargs,
        align_dumper_obj=align_dumper)
    # print(f"model info: ", model)
    input_data = run_torch_modle(model, device="cpu")
    # layer_outputs, _ = get_all_output_tensor(model, device="cpu")
    # node_outputs = run_torch_node_export(model, output_dir="./output/layer_outputs/torch")
    # print(f"node count: {len(node_outputs)}")
    # save_layer_outputs_to_file(node_outputs, "./output/layer_outputs/torch/")
    align_dumper.save()

if __name__ == "__main__":
    main()