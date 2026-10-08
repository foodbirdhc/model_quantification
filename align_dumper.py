import os
import json
import numpy as np
import torch
from collections import OrderedDict


class AlignDumper:
    """
    PyTorch 模型 ALIGN_POINT 中间 Tensor Dump 工具。

    功能：
    1. 注册 ALIGN_POINT
    2. 捕获 Tensor / tuple / list / 嵌套结构
    3. 自动 detach().cpu()
    4. 保存 .npy
    5. 保存 JSON 统计信息
    6. 检测 NaN / Inf
    7. 支持同一个 ALIGN_POINT 多次 capture
    """

    def __init__(
        self,
        output_dir="./align_outputs",
        save_npy=True,
        save_json=True,
        overwrite=True,
    ):
        self.output_dir = output_dir
        self.save_npy = save_npy
        self.save_json = save_json
        self.overwrite = overwrite

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存所有 ALIGN_POINT 的原始数据
        #
        # {
        #     "ALIGN_POINT_001": tensor,
        #     "ALIGN_POINT_002": tensor,
        # }
        self.outputs = OrderedDict()

        # 保存统计信息
        self.statistics = OrderedDict()

        # 注册过的 ALIGN_POINT
        self.registered_points = []

        # 每个 ALIGN_POINT 的 capture 次数
        self.capture_count = {}
        # 当用于生成 ONNX 导出 wrapper 时，启用 export_mode
        # export_mode=True 时，capture 不再 detach 到 cpu，而是保留原始 Tensor（用于成为 graph outputs）
        self.export_mode = False

    # ============================================================
    # 1. 注册 ALIGN_POINT
    # ============================================================

    def register(self, name):
        """
        注册一个 ALIGN_POINT。

        例如：
            dumper.register("ALIGN_POINT_001")
        """

        if not isinstance(name, str):
            raise TypeError("ALIGN_POINT name must be str")

        if not name.startswith("ALIGN_POINT_"):
            raise ValueError(
                f"Invalid ALIGN_POINT name: {name}. "
                f"Expected format: ALIGN_POINT_xxx"
            )

        if name not in self.registered_points:
            self.registered_points.append(name)

        if name not in self.capture_count:
            self.capture_count[name] = 0

    # ============================================================
    # 2. capture
    # ============================================================

    def capture(self, name, output):
        """
        捕获模型中的一个输出。

        支持：

            Tensor

            tuple(
                Tensor,
                Tensor
            )

            list[
                Tensor,
                Tensor
            ]

            嵌套：
            (
                Tensor,
                [
                    Tensor,
                    Tensor
                ]
            )
        """

        # 如果没有提前 register，则自动注册
        if name not in self.registered_points:
            self.register(name)

        # capture 次数
        self.capture_count[name] += 1

        call_id = self.capture_count[name]

        # export_mode: 保留原始 tensor（不 detach），以便作为 graph outputs
        if self.export_mode:
            # Tensor
            if torch.is_tensor(output):
                key = self._make_key(name, call_id)
                self.outputs[key] = output
                return output

            # tuple / list
            if isinstance(output, (tuple, list)):
                for index, item in enumerate(output):
                    key = self._make_key(name, call_id, index)
                    if torch.is_tensor(item):
                        self.outputs[key] = item
                    else:
                        # 非 tensor 项在导出模式下也跳过，但保留占位 None
                        self.outputs[key] = None
                return output

            # 其他类型在导出模式下不支持
            raise TypeError(f"{name}: unsupported export-mode output type: {type(output)}")

        # --------------------------------------------------------
        # 默认行为：detach 到 cpu 并保存（用于运行时 dump）
        # --------------------------------------------------------
        # Tensor
        if torch.is_tensor(output):
            tensor = self._detach_to_cpu(output)
            key = self._make_key(name, call_id)
            self.outputs[key] = tensor
            return output

        # tuple / list
        elif isinstance(output, (tuple, list)):
            tensors = []
            for index, item in enumerate(output):
                tensor = self._extract_tensor(item, path=str(index))
                tensors.append(tensor)
                key = self._make_key(name, call_id, index)
                self.outputs[key] = tensor
            return output

        else:
            raise TypeError(f"{name}: unsupported output type: {type(output)}")

    # ============================================================
    # 3. Tensor detach + CPU
    # ============================================================

    @staticmethod
    def _detach_to_cpu(tensor):

        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Expected Tensor, got {type(tensor)}"
            )

        return tensor.detach().cpu()

    # ============================================================
    # 4. 递归提取 Tensor
    # ============================================================

    def _extract_tensor(self, obj, path=""):

        # Tensor
        if torch.is_tensor(obj):

            return self._detach_to_cpu(obj)

        # tuple
        if isinstance(obj, tuple):

            result = []

            for i, item in enumerate(obj):

                result.append(
                    self._extract_tensor(
                        item,
                        path=f"{path}.{i}"
                    )
                )

            return result

        # list
        if isinstance(obj, list):

            result = []

            for i, item in enumerate(obj):

                result.append(
                    self._extract_tensor(
                        item,
                        path=f"{path}.{i}"
                    )
                )

            return result

        # 不支持 None
        if obj is None:
            return None

        raise TypeError(
            f"Unsupported object at {path}: {type(obj)}"
        )

    # ============================================================
    # 13. 导出时工具：生成 wrapper，使得 forward 返回 (原始输出, *align_tensors)
    # ============================================================

    def enable_export_mode(self):
        self.export_mode = True

    def disable_export_mode(self):
        self.export_mode = False

    def get_export_wrapper(self, model):
        """返回一个 wrapper module 和对应的 align output names 列表。

        wrapper.forward(...) 会执行原 model 的 forward（其内部调用 `ALIGN_POINT`，由本 dumper.capture 捕获），
        并返回 (orig_outputs..., *captured_tensors_in_order)。

        使用示例：
            dumper.enable_export_mode()
            wrapper = dumper.get_export_wrapper(model)
            torch.onnx.export(wrapper, example_inputs, out_onnx_path, output_names=wrapper_output_names)
            dumper.disable_export_mode()

        返回： (wrapper_module, wrapper_output_names)
        """

        dumper = self

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner, dumper_obj):
                super().__init__()
                self.inner = inner
                self.dumper = dumper_obj

            def forward(self, *args, **kwargs):
                # 清理之前的 capture
                self.dumper.clear()
                # 在导出模式下，capture 会保留原始 graph tensor
                out = self.inner(*args, **kwargs)

                # 按 capture 顺序收集 dumper.outputs 的值
                collected = []
                for k, v in self.dumper.outputs.items():
                    # 只收集 tensor（导出模式下应该是 Tensor 或 None 占位）
                    if torch.is_tensor(v):
                        collected.append(v)
                    else:
                        # 非 tensor 占位时插入一个 0-dummy（避免导出报错）
                        collected.append(torch.zeros((), device=args[0].device))

                # 如果原始输出是 tuple/list，则展开，否则直接作为第一个元素
                if isinstance(out, (tuple, list)):
                    return tuple(out) + tuple(collected)
                else:
                    return (out,) + tuple(collected)

        wrapper = _Wrapper(model, dumper)

        # 构造输出名字列表：先用通用名称占位（用户可以替换为实际 final output names），
        # 然后附加 align point 的 key
        # 注意：如果原始模型输出是多个，用户应自行提供实际输出名并替换这里的前缀。
        # 这里仅返回 placeholder names: out_0, out_1, ...
        # 计算原始输出个数 by doing a dry-run? We cannot run model here; assume caller provides names.

        # 注意：align names 只有在 wrapper.forward 被执行一次后才会填充（dumper.outputs 会被 populate）。
        # 因此这里返回 wrapper 与一个 helper callable，用于在 dry-run 后获取 align output keys 列表。
        return wrapper, (lambda: list(self.outputs.keys()))

    # ============================================================
    # 5. 生成 key
    # ============================================================

    @staticmethod
    def _make_key(name, call_id, index=None):

        if index is None:

            if call_id == 1:
                return name

            return f"{name}_call{call_id}"

        else:

            if call_id == 1:
                return f"{name}_{index}"

            return f"{name}_call{call_id}_{index}"

    # ============================================================
    # 6. 保存所有输出
    # ============================================================

    def save(self):

        print("=" * 70)
        print("Saving ALIGN_POINT outputs")
        print("=" * 70)

        for name, tensor in self.outputs.items():

            self._save_tensor(
                name,
                tensor
            )

        # 保存 JSON
        if self.save_json:

            self._save_statistics()

        print("=" * 70)
        print(
            f"ALIGN_POINT dump finished. "
            f"Output directory: {self.output_dir}"
        )
        print("=" * 70)

    # ============================================================
    # 7. 保存 Tensor
    # ============================================================

    def _save_tensor(self, name, tensor):

        if tensor is None:
            return

        # --------------------------------------------------------
        # Tensor
        # --------------------------------------------------------

        if torch.is_tensor(tensor):

            tensor_cpu = tensor.detach().cpu()

            # numpy
            array = tensor_cpu.numpy()

            # 文件名
            filename = os.path.join(
                self.output_dir,
                f"{name}.npy"
            )

            if (
                self.overwrite
                or not os.path.exists(filename)
            ):
                np.save(
                    filename,
                    array
                )

            # 统计信息
            self.statistics[name] = (
                self._calculate_statistics(
                    tensor_cpu
                )
            )

            print(
                f"[SAVE] {name:40s} "
                f"shape={tuple(tensor_cpu.shape)} "
                f"dtype={tensor_cpu.dtype}"
            )

            return

        # --------------------------------------------------------
        # list / tuple
        # --------------------------------------------------------

        if isinstance(tensor, (list, tuple)):

            for i, item in enumerate(tensor):

                self._save_tensor(
                    f"{name}_{i}",
                    item
                )

    # ============================================================
    # 8. 计算 Tensor 统计信息
    # ============================================================

    @staticmethod
    def _calculate_statistics(tensor):

        result = {}

        result["shape"] = list(tensor.shape)

        result["dtype"] = str(tensor.dtype)

        result["numel"] = tensor.numel()

        # 空 Tensor
        if tensor.numel() == 0:

            result["min"] = None
            result["max"] = None
            result["mean"] = None
            result["std"] = None
            result["abs_max"] = None

            result["nan_count"] = 0
            result["inf_count"] = 0

            return result

        # --------------------------------------------------------
        # 对非浮点类型进行统计
        # --------------------------------------------------------

        if not (
            tensor.is_floating_point()
            or tensor.is_complex()
        ):

            stat_tensor = tensor.float()

        else:

            stat_tensor = tensor

        # --------------------------------------------------------
        # NaN
        # --------------------------------------------------------

        if stat_tensor.is_floating_point():

            nan_count = torch.isnan(
                stat_tensor
            ).sum().item()

            inf_count = torch.isinf(
                stat_tensor
            ).sum().item()

        else:

            nan_count = 0
            inf_count = 0

        result["nan_count"] = int(nan_count)

        result["inf_count"] = int(inf_count)

        # --------------------------------------------------------
        # 如果包含 NaN / Inf
        # --------------------------------------------------------

        finite_mask = torch.isfinite(
            stat_tensor
        )

        if finite_mask.any():

            finite_tensor = stat_tensor[
                finite_mask
            ]

            result["min"] = float(
                finite_tensor.min().item()
            )

            result["max"] = float(
                finite_tensor.max().item()
            )

            result["mean"] = float(
                finite_tensor.mean().item()
            )

            result["std"] = float(
                finite_tensor.std(
                    unbiased=False
                ).item()
            )

            result["abs_max"] = float(
                finite_tensor.abs().max().item()
            )

        else:

            result["min"] = None
            result["max"] = None
            result["mean"] = None
            result["std"] = None
            result["abs_max"] = None

        return result

    # ============================================================
    # 9. 保存 JSON
    # ============================================================

    def _save_statistics(self):

        filename = os.path.join(
            self.output_dir,
            "statistics.json"
        )

        data = {
            "registered_points":
                self.registered_points,

            "capture_count":
                self.capture_count,

            "outputs":
                self.statistics,
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"[SAVE] statistics.json"
        )

    # ============================================================
    # 10. 获取某个 ALIGN_POINT
    # ============================================================

    def get(self, name):

        if name not in self.outputs:

            raise KeyError(
                f"ALIGN_POINT not found: {name}"
            )

        return self.outputs[name]

    # ============================================================
    # 11. 清空
    # ============================================================

    def clear(self):

        self.outputs.clear()

        self.statistics.clear()

        for name in self.capture_count:

            self.capture_count[name] = 0

    # ============================================================
    # 12. 打印摘要
    # ============================================================

    def summary(self):

        print()
        print("=" * 70)
        print("ALIGN_POINT Summary")
        print("=" * 70)

        for name, stat in self.statistics.items():

            print(
                f"{name:40s} "
                f"shape={stat['shape']} "
                f"dtype={stat['dtype']} "
                f"min={stat['min']} "
                f"max={stat['max']} "
                f"mean={stat['mean']} "
                f"abs_max={stat['abs_max']}"
            )

        print("=" * 70)