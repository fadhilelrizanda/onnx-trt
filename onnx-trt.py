import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manages CUDA context

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path="model.trt", fp16_mode=True, max_workspace_size=1 << 30):
    """
    Converts an ONNX model to a TensorRT engine and saves it.

    Parameters:
    - onnx_file_path: str - Path to the ONNX file.
    - engine_file_path: str - Path to save the generated TensorRT engine.
    - fp16_mode: bool - Whether to enable FP16 mode for faster inference.
    - max_workspace_size: int - Maximum GPU memory for TensorRT engine.

    Returns:
    - engine: TensorRT engine object.
    """
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Set builder configurations
        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = fp16_mode  # Enable FP16 if supported

        # Load the ONNX model and parse it with TensorRT
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build the TensorRT engine
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_cuda_engine(network)

        if engine:
            # Serialize the engine and save it to file
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("ERROR: Failed to build the TensorRT engine.")

        return engine

# Convert and save the engine
onnx_file = "model.onnx"  # Replace with your ONNX model file path
engine_file = "model.trt"  # Path to save the generated TensorRT engine
build_engine(onnx_file, engine_file)
