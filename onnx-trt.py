import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver automatically

# Select the GPU (e.g., GPU 0)
# cuda.Device(0).make_context()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path="model.trt", use_fp16=True, workspace_size=1 << 30):
    # Initialize the TensorRT builder, network, and parser
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Set up builder configuration
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size  # Set memory pool limit

        # Enable FP16 precision if specified
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build and serialize the engine
        engine = builder.build_serialized_network(network, config)

        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("ERROR: Failed to build the TensorRT engine.")

        return engine

# Usage example
onnx_file = "./dynamic_tsr_dynamic.onnx"  # Replace with your ONNX file path
engine_file = "dynamic_tsr_model.trt"  # File to save the TensorRT engine
build_engine(onnx_file, engine_file)
# cuda.Context.pop()