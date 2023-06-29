import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from colored import stylize, fg
from collections import OrderedDict
import onnxruntime as ort
import ctypes
from cuda import cudart
import os
from colored import stylize, fg
import pycuda.driver as cuda

cuda.init()



now_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(now_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
input_path = os.path.join(data_dir, "input.npy")
output_path = os.path.join(data_dir, "output.npy")
onnx_path = os.path.join(data_dir, "model.onnx")
surgeon_onnx_path = os.path.join(data_dir, "surgeon_model.onnx")
plugin_path = os.path.join(now_dir, "plugin", "librms_norm_plugin.so")
engine_path = os.path.join(data_dir, "model.engine")

batch_size = 1
seq_length = 2048
vocab_size = 64024
embedding_size = 4096


class RMSNorm(torch.nn.Module):
    def __init__(
            self,
            normalized_shape=4096,
            eps=1e-5,
            device=torch.device("cuda"),
            dtype=torch.float32
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.rand(normalized_shape, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt((variance + self.eps))

        return (self.weight * hidden_states).to(input_dtype)


def export_model():
    """
    导出torch模型到onnx
    """
    print("开始导出pytorch模型到onnx")

    # 获取GPU计算结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.rand(
        batch_size, seq_length, embedding_size, dtype=torch.float32).to(device)
    print(input_data.shape)
    print("输入文件保存成功")
    np.save(input_path, input_data.cpu().data.numpy())
    model = RMSNorm().to(device)
    model.eval()
    input_data = input_data.to(device)
    output_tensor = model(input_data)
    output_tensor = output_tensor.cpu().data.numpy().astype(np.float32)
    # print("output is ", outputs)
    print("输出文件保存成功")
    np.save(output_path, output_tensor)

    # 开始导出模型
    torch.onnx.export(
        model,
        args=(input_data,),
        f=onnx_path,
        input_names=["input"],
        output_names=["outputs"],
        opset_version=14,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_length"},
        }
    )
    print("模型导出成功")


def is_rms_norm_node(node: gs.Node):
    """
    判断是否为LayerNorm节点
    """
    if len(node.inputs) == 2:
        left_pre = node.i(0)
        right_pre = node.i(1)
        if left_pre.op != "Cast" or right_pre.op != "Constant":
            return False
        node2 = node.o()
        if node2.op != "ReduceMean":
            return False
        node3 = node2.o()
        if node3.op != "Add" or node3.i(1).op != "Constant":
            return False
        node4 = node3.o()
        if node4.op != "Sqrt":
            return False
        node5 = node4.o()
        if node5.op != "Div":
            return False
        node6 = node5.o()
        if node6.op != "Mul" or node6.i(0) != left_pre:
            return False
        node7 = node6.o()
        if node7.op != "Mul":
            return False
        node8 = node7.o()
        if node8.op != "Cast":
            return False
        return True
    return False


def replace_onnx():
    """
    修改onnx模型，将中间的LayerNorm层替换为TensorRT插件层
    """
    model = onnx.load(onnx_path)
    # 遍历模型的所有初始化张量
    for initializer in model.graph.initializer:
        name = initializer.name
        tensor = initializer.float_data if len(
            initializer.float_data) > 0 else initializer.raw_data
        shape = tuple(dim for dim in initializer.dims)

        # 在这里进行你想要的操作，比如打印权重名称、形状和数值等
        print(f"Name: {name}")
        print(f"Shape: {shape}")
        print(f"Values: {tensor}")
        value = np.frombuffer(initializer.raw_data)
        print(value)
    graph = gs.import_onnx(model)
    n = 0
    for node in graph.nodes:
        if node.op != "Pow":
            continue
        # 判断当前Div是否为RMSNorm
        if is_rms_norm_node(node):
            print("find one layer norm")
            # 属性必须为数组
            # epsilon = node.o().o().inputs[1].values
            epsilon = 1e-5
            # 新增获取gamma, beta属性

            gamma = node.o().o().o().o().o().o().inputs[0].values
            print("epsilon", epsilon)
            print("gamma", gamma)
            temp_outputs = gs.Variable(
                name=f"LayerNorm_output_{n}", dtype=np.float32,
                shape=None
            )
            new_node = gs.Node(
                op="RMSNorm",
                name=f"RMSNorm_{n}",
                attrs=OrderedDict(epsilon=epsilon, gamma=gamma),
                inputs=[node.i(0).inputs[0]],
                outputs=[temp_outputs]
            )
            graph.nodes.append(new_node)
            out_node = node.o().o().o().o().o().o().o()
            print("output node is ", out_node)
            # 建立连接
            if len(out_node.outputs) > 0 and \
                    len(out_node.outputs[0].outputs) > 0:
                # out_node.o().inputs[0] = temp_outputs
                out_name = out_node.outputs[0]
                # print("out_name", out_name)
                for sub_node in list(out_node.outputs[0].outputs):
                    # print("sub node", sub_node)
                    for i, input_node in enumerate(sub_node.inputs):
                        if input_node == out_name:
                            print("link node ", i, input_node)
                            sub_node.inputs[i] = temp_outputs
            # 最后一个节点
            else:
                new_node.outputs = out_node.outputs
                # raise Exception("请检查节点")
            out_node.outputs.clear()
            n += 1
    graph.cleanup()
    print(graph.outputs)
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print("surgeon onnx model save success!")


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            print(stylize("[ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.WARNING:
            print(stylize("[WARNING] " + msg, fg("yellow")))  # 黄色字体
        elif severity == trt.Logger.INTERNAL_ERROR:
            print(stylize("[INTERNAL_ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.INFO:
            print(stylize("[INFO] " + msg, fg("green")))  # 绿色字体
        elif severity == trt.Logger.VERBOSE:
            print(stylize("[VERBOSE] " + msg, fg("blue")))  # 蓝色字体
        else:
            print("[UNKNOWN] " + msg)


def build_engine(onnx_path1: str):
    """
    构建TensorRT引擎
    """
    print("=== build TRT engine ===")
    # logger = trt.Logger(trt.Logger.INFO)
    logger = MyLogger()
    if os.path.exists(plugin_path):
        ctypes.cdll.LoadLibrary(plugin_path)
        print("plugin path is", plugin_path)
    else:
        raise Exception("plugin not found")
    print(ctypes.get_errno())
    trt.init_libnvinfer_plugins(logger, "")
    # registry = trt.get_plugin_registry()
    # creator = registry.get_plugin_creator('RMSNorm', '1')
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # disable tf32
    config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
    print("use fp16 ?", config.get_flag(trt.BuilderFlag.FP16))
    profile = builder.create_optimization_profile()
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 解析onnx
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path1, "rb") as f:
        parser.parse(f.read())
        for i in range(parser.num_errors):
            print(stylize("parser erorr ", fg("red")), parser.get_error(i))
    input_tensor = network.get_input(0)
    print("input name", input_tensor.name)
    print("output name", network.get_output(0).name)
    profile.set_shape(
        input_tensor.name,
        min=[1, 1, embedding_size],
        opt=[batch_size, seq_length, embedding_size],
        max=[batch_size * 2, seq_length * 2, embedding_size]
    )
    config.add_optimization_profile(profile)
    engine_data = builder.build_serialized_network(network, config)
    if engine_data is None:
        print(stylize("engine data build failed", fg("red")))
        raise Exception("engine data build failed")
    with open(engine_path, "wb") as f:
        f.write(engine_data)
        print(stylize("build TensorRT Engine success!", fg("yellow")))


def gpu_check(error: cudart.cudaError_t):
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)
        error_info = cudart.cudaGetErrorString(error)
        print(stylize(f"ERROR [{error_name}]: {error_info}", fg("red")))
        raise Exception(f"ERROR [{error_name}]: {error_info}")


def inference():
    """
    正式对模型进行推理
    """
    logger = MyLogger()
    if os.path.exists(plugin_path):
        ctypes.cdll.LoadLibrary(plugin_path)
        print("plugin path is", plugin_path)
    else:
        raise Exception("plugin not found")
    print(ctypes.get_errno())
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    (error_info, stream) = cudart.cudaStreamCreate()
    gpu_check(error_info)
    input_name = engine.get_tensor_name(0)
    context.set_input_shape(input_name, [batch_size, seq_length, embedding_size])
    print("=" * 20, "inference", "=" * 20)
    h_input = np.ascontiguousarray(
        np.load(input_path).reshape(-1)).astype(np.float32)
    output_name = engine.get_tensor_name(1)
    output_shape = context.get_tensor_shape(output_name)
    h_output = np.empty(
        shape=output_shape,
        dtype=trt.nptype(engine.get_tensor_dtype(output_name))
    )
    print("output shape", h_output.shape)
    print("output type", h_output.dtype)

    error_info, d_input = cudart.cudaMallocAsync(h_input.nbytes, stream)
    gpu_check(error_info)
    error_info, d_output = cudart.cudaMallocAsync(h_output.nbytes, stream)
    gpu_check(error_info)
    (error_info, ) = cudart.cudaMemcpyAsync(
        d_input,
        h_input.ctypes.data,
        h_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        stream
    )
    gpu_check(error_info)
    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)
    context.execute_async_v3(stream)
    (error_info, ) = cudart.cudaMemcpyAsync(
        h_output.ctypes.data,
        d_output,
        h_output.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )
    gpu_check(error_info)
    (error_info, ) = cudart.cudaStreamSynchronize(stream)
    gpu_check(error_info)
    (error_info, ) = cudart.cudaDeviceSynchronize()
    gpu_check(error_info)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    return h_output


def check(a, b, weak=False, checkEpsilon=1e-4):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:", res, diff0, diff1)


if __name__ == "__main__":
    # 1. 导出模型
    export_model()
    # 2. 替换模型
    replace_onnx()
    # 3. 构建TensorRT Engine
    build_engine(surgeon_onnx_path)
    # build_engine(onnx_path)
    # 4. 获取推理结果
    outputs2_1 = inference()
    print(outputs2_1)
    # 5. 对比计算结果
    outputs1_1 = np.load(output_path)
    print("=" * 20, "Pytorch VS TensorRT", "=" * 20)
    check(outputs2_1, outputs1_1, True)
    print("=" * 50)

    """
    print("onnx runtime result")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = np.load(input_path)
    outputs = session.run(
        ["outputs"], {"input": inputs})
    print("*" * 20)
    print("=" * 20, "Pytorch VS ONNX", "=" * 20)
    check(outputs[0], outputs1_1, True)
    print("*" * 20)
    """
