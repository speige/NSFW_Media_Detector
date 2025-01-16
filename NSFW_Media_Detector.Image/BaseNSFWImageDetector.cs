using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace NSFW_Media_Detector.Image
{
    public interface INSFWImageDetector: IDisposable
    {
        public float CalcNSFWProbability(byte[] imageBytes);
        public float CalcNSFWProbability(Image<Rgba32> image);
    }

    public abstract class BaseNSFWImageDetector<T> : INSFWImageDetector
    {
        protected InferenceSession _session;
        protected readonly string _inputTensorName;
        protected readonly string _outputTensorName;
        protected readonly int _resizeWidth;
        protected readonly int _resizeHeight;
        protected Func<float, float> _pixelTransformer { get; init; }
        protected Func<float, float> _pixelTransformer_R { get; init; }
        protected Func<float, float> _pixelTransformer_G { get; init; }
        protected Func<float, float> _pixelTransformer_B { get; init; }
        protected int[] _shape;
        protected bool _padToMaintainAspectRatio = false;
        protected IResampler _resizeSampler = KnownResamplers.Bicubic;
        protected bool _swapRedBlue = false;
        protected bool _tensorLayoutChannelFirst = false;

        public BaseNSFWImageDetector(string modelPath, string inputTensorName, string outputTensorName, int resizeWidth, int resizeHeight)
        {
            _inputTensorName = inputTensorName;
            _outputTensorName = outputTensorName;
            _resizeWidth = resizeWidth;
            _resizeHeight = resizeHeight;
            _pixelTransformer = x => x;

            _shape = new int[] { 1, _resizeHeight, _resizeWidth, 3 };

            var sessionOptions = new SessionOptions()
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                //LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
            };
            for (var deviceId = 0; deviceId <= 2; deviceId++)
            {
                try
                {
                    sessionOptions.AppendExecutionProvider_DML(deviceId);
                    break;
                }
                catch (Exception e)
                {
                    if (e.Message.Contains("handle is invalid", StringComparison.InvariantCultureIgnoreCase))
                    {
                        break;
                    }

                    Console.WriteLine("Unable to use GPU Acceleration: " + e.Message);
                }
            }
            sessionOptions.AppendExecutionProvider_CPU();

            _session = new InferenceSession(ReadFileChunked(modelPath), sessionOptions);

            //var suggestedInputName = _session.InputNames[0];
            //var suggestedOutputName = _session.OutputNames[0];
            //var suggestedShape = _session.InputMetadata[inputTensorName].Dimensions;
        }

        public static byte[] ReadFileChunked(string modelPath)
        {
            List<byte> result = new List<byte>();
            var i = 0;
            while (true)
            {
                var fileName = $"{modelPath}.{i}";
                if (!File.Exists(fileName))
                {
                    break;
                }

                result.AddRange(File.ReadAllBytes(fileName));
                i++;
            }

            return result.ToArray();
        }

        public static void SplitFileIntoChunks(string modelPath, int chunkSizeInMB = 50)
        {
            var chunks = File.ReadAllBytes(modelPath).Chunk(chunkSizeInMB * 1024 * 1024).ToList();
            for (var i = 0; i < chunks.Count; i++)
            {
                File.WriteAllBytes($"{modelPath}.{i}", chunks[i]);
            }
        }

        protected class PreprocessedImage : IDisposable
        {
            public Image<Rgba32> Image;
            public float[] PixelData;
            public float RatioX;
            public float RatioY;
            public int XPadding;
            public int YPadding;
            public int OriginalWidth;
            public int OriginalHeight;

            public void Dispose()
            {
                if (Image != null)
                {
                    Image.Dispose();
                }
            }
        }

        protected PreprocessedImage PreprocessImage(Image<Rgba32> image)
        {
            var result = new PreprocessedImage();
            result.Image = image.Clone();
            result.OriginalWidth = result.Image.Width;
            result.OriginalHeight = result.Image.Height;
            result.RatioX = (float)_resizeWidth / result.Image.Width;
            result.RatioY = (float)_resizeHeight / result.Image.Height;
            result.Image.Mutate(ctx => ctx.Resize(new ResizeOptions() { Mode = _padToMaintainAspectRatio ? ResizeMode.Pad : ResizeMode.Stretch, Size = new Size(_resizeWidth, _resizeHeight), Sampler = _resizeSampler, PadColor = Color.Black }));
            result.PixelData = GetPixelData(result);
            return result;
        }

        protected float[] GetPixelData(PreprocessedImage preprocessed)
        {
            int totalPixels = preprocessed.Image.Width * preprocessed.Image.Height;
            float[] pixelData = new float[totalPixels * 3];
            preprocessed.Image.ProcessPixelRows(pixels => {
                var idx = 0;
                for (int y = 0; y < preprocessed.Image.Height; y++)
                {
                    var rowSpan = pixels.GetRowSpan(y);
                    for (int x = 0; x < preprocessed.Image.Width; x++)
                    {
                        Rgba32 c = rowSpan[x];

                        float red = c.R;
                        float green = c.G;
                        float blue = c.B;

                        red = (_pixelTransformer_R ?? _pixelTransformer)(red);
                        green = (_pixelTransformer_G ?? _pixelTransformer)(green);
                        blue = (_pixelTransformer_B ?? _pixelTransformer)(blue);

                        if (_tensorLayoutChannelFirst)
                        {
                            int baseOffset = y * preprocessed.Image.Width + x;

                            pixelData[baseOffset] = _swapRedBlue ? blue : red;
                            pixelData[totalPixels + baseOffset] = green;
                            pixelData[2 * totalPixels + baseOffset] = _swapRedBlue ? red : blue;
                        }
                        else
                        {
                            pixelData[idx + 0] = _swapRedBlue ? blue : red;
                            pixelData[idx + 1] = green;
                            pixelData[idx + 2] = _swapRedBlue ? red : blue;
                        }

                        idx += 3;
                    }
                }
            });

            return pixelData;
        }

        protected virtual Tensor<float> ImageToTensor(PreprocessedImage image)
        {
            return new DenseTensor<float>(image.PixelData, _shape);
        }

        protected TensorBase RunModel<T>(Tensor<T> input, string inputTensorName, string outputTensorName)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputTensorName, input) };

            using (var results = _session.Run(inputs))
            {
                var output = results.FirstOrDefault(r => r.Name == outputTensorName);
                return output?.Value as TensorBase;
            }
        }

        protected virtual TensorBase RunModel(PreprocessedImage image)
        {
            return RunModel(ImageToTensor(image), _inputTensorName, _outputTensorName);
        }

        public void Dispose()
        {
            _session.Dispose();
        }

        protected abstract T ModelOutputToDetections(PreprocessedImage preprocessed, TensorBase modelOutput);

        public virtual T Detect(byte[] imageBytes)
        {
            using (var memoryStream = new MemoryStream(imageBytes))
            using (var image = SixLabors.ImageSharp.Image.Load<Rgba32>(memoryStream))
            {
                return Detect(image);
            }
        }

        private bool _initialized = false;
        public virtual T Detect(Image<Rgba32> image)
        {
            if (!_initialized)
            {
                //NOTE: Ocassionally a model gets garbage results when ran the 1st time. Seems to be a bug in ONNX.
                _initialized = true;
                var throwAway = Detect(image);
            }

            using (var preprocessed = PreprocessImage(image))
            {
                var output = RunModel(preprocessed);
                return ModelOutputToDetections(preprocessed, output);
            }
        }


        protected abstract float DetectionToProbability(T detection);

        public virtual float CalcNSFWProbability(byte[] imageBytes)
        {
            return (float)Math.Round(DetectionToProbability(Detect(imageBytes)), 2);
        }

        public virtual float CalcNSFWProbability(Image<Rgba32> image)
        {
            return (float)Math.Round(DetectionToProbability(Detect(image)), 2);
        }

        protected float[] Softmax(DenseTensor<float> logits)
        {
            float maxLogit = logits.Max();
            float[] expLogits = logits.Select(x => (float)Math.Exp(x - maxLogit)).ToArray();
            float sumExpLogits = expLogits.Sum();
            return expLogits.Select(x => x / sumExpLogits).ToArray();
        }
    }
}