Must be in ONNX format

HuggingFace can be converted with optimum-cli tool, instructions at https://huggingface.co/docs/transformers/en/serialization

optimum-cli doesn't support all formats. FocalNet can be exported via PyTorch torch.onnx.export() function

TensorFlow can be converted with this tool https://github.com/onnx/tensorflow-onnx

Run through BaseDetector.SplitFileIntoChunks to avoid > 100MB error with GitHub