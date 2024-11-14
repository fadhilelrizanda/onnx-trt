import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver automatically
cuda.Device(0).make_context()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path="model.trt", use_fp16=True, workspace_size=1 << 30):
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

        # Add an optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            input_shape = network.get_input(i).shape
            if -1 in input_shape:  # Check if there's a dynamic dimension
                # Set dynamic shape ranges (e.g., batch size 1 to 5, or other dimensions as needed)
                profile.set_shape(input_name, (1, 3, 416, 416), (3, 3, 416, 416), (5, 3, 416, 416))
        config.add_optimization_profile(profile)

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