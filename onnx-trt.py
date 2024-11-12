import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver automatically
cuda.Device(0).make_context()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path="model.trt", use_fp16=True, workspace_size=1 << 30):
    """
    Converts an ONNX model to a TensorRT engine and saves it to a file.

    Args:
        onnx_file_path (str): Path to the ONNX model file.
        engine_file_path (str): Path to save the generated TensorRT engine.
        use_fp16 (bool): Whether to enable FP16 precision mode.
        workspace_size (int): Workspace memory size for TensorRT.

    Returns:
        engine: A TensorRT engine object, or None if conversion failed.
    """
    # Initialize the TensorRT builder, network, and parser
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Set up builder configuration
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)  # Set memory pool limit
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision if supported

        # Parse the ONNX model file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build and save the engine
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_serialized_network(network, config)

        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("ERROR: Failed to build the TensorRT engine.")

        return engine

# Usage example
onnx_file = "./test_static.onnx"  # Replace with your ONNX file path
engine_file = "model.trt"  # File to save the TensorRT engine
build_engine(onnx_file, engine_file)
cuda.Context.pop()