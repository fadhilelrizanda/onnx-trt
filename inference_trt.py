import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver automatically
import numpy as np
import cv2

# Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Load a serialized engine from file."""
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Allocates host and device buffers for TensorRT engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """Perform inference with TensorRT."""
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs
    return [out["host"] for out in outputs]

def preprocess_image(image_path, input_shape):
    """Preprocess the image for model inference."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[2], input_shape[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize if required
    image = np.transpose(image, (2, 0, 1))  # Channel first
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the TensorRT engine
engine_file = "model.trt"
engine = load_engine(engine_file)

# Create execution context
context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Set the input image
input_image = preprocess_image("./data/prediksi4.jpg", engine.get_binding_shape(0))  # Replace with your image file path
np.copyto(inputs[0]["host"], input_image.ravel())

# Perform inference
output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

# Postprocess output (example assumes a single output layer and classification task)
predictions = np.array(output[0]).reshape(engine.get_binding_shape(1))
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
