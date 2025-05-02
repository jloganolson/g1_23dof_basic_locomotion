import onnxruntime as rt

# Get the list of available execution providers
providers = rt.get_available_providers()
print(f"Available Execution Providers: {providers}")

# If CUDA is properly configured and the onnxruntime-gpu package is correctly installed,
# you should see 'CUDAExecutionProvider' in the list.

# You can also try to explicitly request the CUDA execution provider when creating an InferenceSession:
try:
    ort_session = rt.InferenceSession("bh_policy.onnx", providers=['CUDAExecutionProvider'])
    print("CUDA Execution Provider is available and session created successfully.")
except Exception as e:
    print(f"Error creating CUDA Inference Session: {e}")
    print("Make sure CUDA is properly installed and configured.")

# To further verify during inference, you can try running a model and monitor
# your GPU usage using tools like nvidia-smi (on Linux and Windows).
# If the GPU utilization increases during the inference, it indicates that CUDA is being used.

# Example of running inference (assuming you have a model loaded):
# ort_inputs = {ort_session.get_inputs()[0].name: your_input_data}
# ort_outputs = ort_session.run(None, ort_inputs)