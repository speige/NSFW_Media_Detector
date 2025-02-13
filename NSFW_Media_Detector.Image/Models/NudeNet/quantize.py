import onnx
import onnxoptimizer
import onnxruntime
import onnxruntime.quantization as ortq
import glob
import cv2
import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnxoptimizer
from onnx.tools import update_model_dims
import logging

# NOTE: quantization gives worse accuracy for a smaller model with possible performance improvement, however in testing, despite the model size shrinking, the performance actually got worse

# quantization requires example images for determining activation ranges. Example dataset here (use extreme caution): https://universe.roboflow.com/usman-ixf1b/nudity-noeag/dataset/2

# Enable debug-level logging
logging.basicConfig(level=logging.DEBUG)

# Paths
onnx_model_path = "C:/temp/quantize/640m.onnx"
optimized_model_path = "C:/temp/quantize/640m_optimized.onnx"
optimized_inferred_model_path = "C:/temp/quantize/640m_optimized_inferred.onnx"
quantized_model_path = "C:/temp/quantize/640m_INT8.onnx"
image_dir = "C:/temp/quantize/nudity.v2i.coco/**/*.jpg"  # Recursively search for images

# Step 1: Define Preprocessing Function
def preprocess_image(image_path, input_shape=(1, 3, 640, 640)):
    """Preprocess an image for YOLO static quantization."""
    
    # Load image in BGR format
    img = cv2.imread(image_path).astype(np.float32)

    # Resize while maintaining aspect ratio
    target_size = input_shape[2]
    scale = min(target_size / img.shape[1], target_size / img.shape[0])
    new_w, new_h = round(img.shape[1] * scale), round(img.shape[0] * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Apply Centered Padding
    pad_h_top = round((target_size - new_h) / 2)
    pad_h_bottom = target_size - new_h - pad_h_top
    pad_w_left = (target_size - new_w) // 2
    pad_w_right = target_size - new_w - pad_w_left

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Swap Red and Blue channels
    img_padded = img_padded[..., ::-1]  # Convert BGR → RGB

    # Normalize
    img_padded /= 255.0

    # Convert to Channel-First format (C, H, W)
    img_padded = np.transpose(img_padded, (2, 0, 1))

    # Add batch dimension (1, 3, 640, 640)
    return np.expand_dims(img_padded, axis=0)

# Step 2: Define Calibration Data Reader
class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name):
        """Reads images and prepares them as calibration data."""
        self.image_paths = glob.glob(image_dir, recursive=True)
        self.input_name = input_name
        self.data_iter = iter(self.image_paths)

    def get_next(self):
        """Load the next batch of images for calibration."""
        try:
            image_path = next(self.data_iter)
            input_tensor = preprocess_image(image_path)
            return {self.input_name: input_tensor}
        except StopIteration:
            return None

    def rewind(self):
        """Restart the iterator if needed."""
        self.data_iter = iter(self.image_paths)

# Step 3: Load Model and Get Input Name
onnx_model = onnx.load(onnx_model_path)

# doesn't have any noticeable affect on accuracy or speed
# Step 2: Apply standard ONNX graph optimizations (best for YOLO)
# passes = [
#     "eliminate_deadend",  # Remove unnecessary branches
#     "eliminate_identity",  # Remove identity ops
#     "eliminate_nop_dropout",  # Remove unused dropout layers
#     "eliminate_nop_pad",  # Remove unnecessary padding
#     "eliminate_nop_transpose",  # Remove useless transposes
#     "eliminate_unused_initializer",  # Remove unused weights
#     "fuse_bn_into_conv",  # Fuse batch normalization into convolutions
#     "fuse_consecutive_squeezes",  # Optimize squeeze operations
# ]
# 
# optimized_model = onnxoptimizer.optimize(onnx_model, passes)
# onnx.save(optimized_model, optimized_model_path)

# doesn't have any noticeable affect on accuracy or speed
# onnx.shape_inference.infer_shapes(optimized_model)
# onnx.save(optimized_model, optimized_inferred_model_path)

session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name   # Extract input tensor name

# Step 4: Run Static Quantization with Calibration Data
data_reader = ImageDataReader(image_dir, input_name)

ortq.quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=data_reader,
    quant_format=ortq.QuantFormat.QDQ,  # Quantize using QDQ format for accuracy
    weight_type=ortq.QuantType.QInt8,  # Convert weights to INT8
    activation_type=ortq.QuantType.QInt8,  # Convert activations to INT8
    op_types_to_quantize=["Conv", "MatMul", "Add", "Mul", "Div", "Sub", "Reshape", "Transpose", "Split", "Sigmoid", "Softmax", "MaxPool", "Resize" ] #
)

print(f"Fully Quantized (INT8) ONNX model saved to: {quantized_model_path}")
