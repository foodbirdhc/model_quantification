import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
import json

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

def save_layer_outputs_to_file(layer_outputs, output_dir):
    """
    Save the layer outputs to files in the specified directory.
    Args:
        layer_outputs (dict): A dictionary containing layer outputs.
        output_dir (str): Directory to save the output files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for tensor_name, tensor in layer_outputs.items():
        tensor_npy_file_path = output_path / f"npy/{tensor_name}.npy"
        tensor_json_file_path = output_path / f"json/{tensor_name}.json"

        tensor_npy_file_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_json_file_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(tensor_npy_file_path, tensor)

        # result_dict =  {
        #     "layer_name": tensor_name,
        #     "shape": tensor.shape,
        #     "dtype": str(tensor.dtype),
        #     "data": tensor.tolist()
        # }
        result_dict =  {
            "layer_name": tensor_name,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
            "nan_count": int(np.isnan(tensor).sum()),
            "inf_count": int(np.isinf(tensor).sum()),
        }
        with open(tensor_json_file_path, "w") as f:
            json.dump(result_dict, f, indent=2)

def auto_detect_module_prefixes(model):
    """启发式自动检测 ONNX 中包含模块语义的前缀。"""
    prefix_counter = {}

    for node in model.graph.node:
        # 优先看 node.name，因为它更接近 PyTorch 的模块名
        if node.name:
            name = node.name.strip()
            # 取首段前缀，如 "encoders.line.layers.0" -> "encoders"
            if "." in name:
                prefix = name.split(".", 1)[0]
                prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1

        for out_name in node.output:
            if not out_name:
                continue
            if "." in out_name:
                prefix = out_name.split(".", 1)[0]
                prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1

    # 只保留高频、较有意义的前缀，避免把泛化前缀当成模块名
    candidate_prefixes = []
    for prefix, count in sorted(prefix_counter.items(), key=lambda x: x[1], reverse=True):
        if count >= 2 and len(prefix) > 2:
            candidate_prefixes.append(prefix + ".")

    # 额外兜底：常见模块名关键词
    fallback_prefixes = [
        "encoders.",
        "transformer.",
        "gate_fusion",
        "initial_decoder.",
        "ln",
        "pos_emb",
        "final_",
    ]
    for prefix in fallback_prefixes:
        if prefix not in candidate_prefixes:
            candidate_prefixes.append(prefix)

    return tuple(candidate_prefixes)


def convert_model_to_module_output(model_path, output_model_path,
                                  module_prefixes=None,
                                  keep_existing_outputs=True):
    model = onnx.load(model_path)
    graph = model.graph

    if module_prefixes is None:
        module_prefixes = auto_detect_module_prefixes(model)

    value_info_map = {}
    for vi in graph.value_info:
        value_info_map[vi.name] = vi
    for inp in graph.input:
        value_info_map[inp.name] = inp
    for out in graph.output:
        value_info_map[out.name] = out

    original_outputs = [o for o in graph.output]
    matched_outputs = []
    seen = set()

    for node in graph.node:
        node_name = (node.name or "").strip()
        for out_name in node.output:
            if not out_name:
                continue

            candidate = out_name
            if node_name and any(node_name.startswith(prefix) for prefix in module_prefixes):
                candidate = node_name

            if not any(candidate.startswith(prefix) for prefix in module_prefixes):
                continue

            if candidate in seen:
                continue
            seen.add(candidate)

            value_info = value_info_map.get(out_name)
            if value_info is None:
                continue

            matched_outputs.append(
                onnx.helper.make_tensor_value_info(
                    candidate,
                    value_info.type.tensor_type.elem_type,
                    value_info.type.tensor_type.shape
                )
            )

    # 如果没有匹配到模块输出，保留原始输出并写出到目标文件，避免只剩最终输出
    if not matched_outputs:
        print("[module-output] no module-like names matched; copy original model to output path.")
        output_parent = Path(output_model_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_model_path)
        print(f"[module-output] saved original model to: {output_model_path}")
        return output_model_path

    graph.output.clear()
    for out in matched_outputs:
        graph.output.append(out)

    if keep_existing_outputs:
        existing_names = {o.name for o in graph.output}
        for out in original_outputs:
            if out.name not in existing_names:
                graph.output.append(out)

    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_model_path)
    print(f"[module-output] saved to: {output_model_path}")
    print(f"[module-output] matched output count: {len(matched_outputs)}")
    for out in graph.output:
        print("  -", out.name)
    return output_model_path

def convert_model_to_all_node_output(model_path, output_model_path):
    """
    Convert the ONNX model to output all intermediate node outputs.
    Args:
        model_path (str): Path to the original ONNX model file.
        output_model_path (str): Path to save the modified ONNX model file.
    """
    model = onnx.load(model_path)
    # 推断中间 tensor 的 shape 和 dtype
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed: {e}")

    graph = model.graph

    # 已经注册为 graph output 的 tensor 名称
    existing_output_names = {
        output.name
        for output in model.graph.output
    }

    print(f"Existing output names: {existing_output_names}")

    value_info_map = {}

    for value_info in graph.value_info:
        value_info_map[value_info.name] = value_info
    
    for graph_input in graph.input:
        value_info_map[graph_input.name] = graph_input
    
    for graph_output in graph.output:
        value_info_map[graph_output.name] = graph_output
    
    added_output_names = []

    for node in graph.node:
        for tensor_name in node.output:
            if not tensor_name:
                continue
            
            if tensor_name in existing_output_names:
                print(f"Tensor '{tensor_name}' is already an output. Skipping.")
                continue
            
            value_info = value_info_map.get(tensor_name)
            
            if value_info is None:
                print(f"Warning: No value_info found for tensor '{tensor_name}'. Skipping.")
                continue
            
            graph.output.append(value_info)
            existing_output_names.add(tensor_name)
            added_output_names.append(tensor_name)
    
    output_parent = Path(output_model_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    onnx.save(model, output_model_path)

    print(f"Original output count: {len(existing_output_names) - len(added_output_names)}")
    print(f"Added output count: {len(added_output_names)}")
    print(f"Converted model saved to: {output_model_path}")

    return output_model_path

def get_all_output_tensor(model_path):
    """
    Get the output information from the ONNX model file.
    Args:
        model_path (str): Path to the ONNX model file.
    Returns:
        dict: A dictionary containing the output information.
    """
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3  # Suppress info and warning messages
    session = ort.InferenceSession(
        model_path,
        sess_options=session_opts,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_info = {}
    for input in session.get_inputs():
        input_info[input.name] = {
            "shape": input.shape,
            "dtype": input.type
        }
    output_info = {}
    for output in session.get_outputs():
        output_info[output.name] = {
            "shape": output.shape,
            "dtype": output.type
        }

    print("input_info: ", input_info)
    print("output_info: ", output_info)

    input_feed = {}
    np.random.seed(5)  # For reproducibility
    for input_name, input_shape in zip(INPUT_NAMES, DUMMY_SHAPES):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_feed[input_name] = input_data

        print(
            f"Generated input: "
            f"name={input_name}, "
            f"shape={input_data.shape}, "
            f"dtype={input_data.dtype}"
        )
    print(f"input shape: {[input_data.shape for input_data in input_feed.values()]}")
    # print(f"input_feed: {input_feed}")

    # runtime, None 表示获取所有输出
    outputs = session.run(
        None,
        input_feed=input_feed)

    output_metas = session.get_outputs()

    print(f"outputs shape: {[output.shape for output in outputs]}")
    print(f"output_metas shape: {[output.shape for output in output_metas]}")
    
    all_outputs = {}

    print("Model outputs:")
    print(f"outputs length: {len(outputs)}")
    for output_name, output_value in zip(output_metas, outputs):
        all_outputs[output_name.name] = output_value
        # print(
        #     f"name={output_name}, "
        #     f"shape={output_value.shape}, "
        #     f"dtype={output_value.dtype}, "
        #     f"data={output_value}"
        # )

    return all_outputs

def get_model_info(model_path):
    """
    Get the model information from the ONNX model file.
    Args:
        model_path (str): Path to the ONNX model file.
    Returns:
        dict: A dictionary containing the model information.
    """
    model = onnx.load(model_path)
    model_info = {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": model.model_version,
        "doc_string": model.doc_string,
        "graph_name": model.graph.name,
        "graph_inputs": [(input.name, input.type.tensor_type.elem_type, input.type.tensor_type.shape) for input in model.graph.input],
        "graph_outputs": [(output.name, output.type.tensor_type.elem_type, output.type.tensor_type.shape) for output in model.graph.output],
        "graph_nodes": len(model.graph.node),
    }
    print(f"Model Information for {model_path}:")
    for input_tesor in model.graph.input:
        print("name : ", input_tesor.name)

        shape = []
        for dim in input_tesor.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append(1)
        print("shape: ", shape)
    # for key, value in model_info.items():
    #     print(f"  {key}: {value}")
    return model_info

def main():
    ap = argparse.ArgumentParser(description="ONNX model inference example")
    ap.add_argument(
        "-m", "--model",
        # required=True,
        default="/home/voyah/workspace/data/model/intermediate_results/output_model_quantization/trt_ptq/Aug20_10-18-28/model.onnx",
        help="Path to the ONNX model file.")
    
    args = ap.parse_args()

    # get_model_info(args.model)
    new_model_path = convert_model_to_all_node_output(args.model,"./output/model_all_output.onnx")
    # new_model_path = convert_model_to_module_output(
    #     args.model,
    #     "./output/model_all_output.onnx",
    #     module_prefixes=None,
    # )
    all_layer_outputs = get_all_output_tensor("./output/model_all_output.onnx")
    save_layer_outputs_to_file(all_layer_outputs, "./output/layer_outputs/onnx/")

if __name__ == "__main__":
    main()