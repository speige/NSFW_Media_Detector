using Emgu.CV.Dnn;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Rectangle = System.Drawing.Rectangle;

namespace NSFW_Media_Detector.Image.Models.NudeNet
{
    // https://github.com/notAI-tech/NudeNet
    // https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx

    public class NudeNetDetector : BaseNSFWImageDetector<List<NudeNetDetector.Detection>>
    {
        protected readonly string[] _labels;
        protected readonly Dictionary<string, float> _labelWeights;
        public NudeNetDetector()
            : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\NudeNet\640m.onnx"), "images", "output0", 640, 640)
        {
            _shape = new int[] { 1, 3, _resizeHeight, _resizeWidth };
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = true;
            _tensorLayoutChannelFirst = true;

            var labelsWithWeights = new (string label, float weight)[]
            {
                ("FEMALE_GENITALIA_COVERED", 0.4f),
                ("FACE_FEMALE", 0f),
                ("BUTTOCKS_EXPOSED", 0.75f),
                ("FEMALE_BREAST_EXPOSED", 0.75f),
                ("FEMALE_GENITALIA_EXPOSED", 1f),
                ("MALE_BREAST_EXPOSED", 0.05f),
                ("ANUS_EXPOSED", 1f),
                ("FEET_EXPOSED", 0f),
                ("BELLY_COVERED", 0f),
                ("FEET_COVERED", 0f),
                ("ARMPITS_COVERED", 0f),
                ("ARMPITS_EXPOSED", 0.05f),
                ("FACE_MALE", 0f),
                ("BELLY_EXPOSED", 0.15f),
                ("MALE_GENITALIA_EXPOSED", 1f),
                ("ANUS_COVERED", 0.5f),
                ("FEMALE_BREAST_COVERED", 0.30f),
                ("BUTTOCKS_COVERED", 0.15f)
            };

            _labels = labelsWithWeights.Select(x => x.label).ToArray();
            _labelWeights = labelsWithWeights.ToDictionary(x => x.label, x => x.weight);
        }

        public class Detection
        {
            public string Class { get; set; }
            public float Score { get; set; }
            public int[] Box { get; set; }
        }

        protected override float DetectionToProbability(List<Detection> detection)
        {
            return (float)Math.Clamp(Math.Round(detection.Select(x => _labelWeights[x.Class] * x.Score).Sum(), 2), 0, 1);
        }

        protected override List<Detection> ModelOutputToDetections(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            var output = (DenseTensor<float>)modelOutput;
            if (output.Dimensions.Length != 3 || output.Dimensions[0] != 1)
            {
                throw new ArgumentException("Output tensor must have shape [1, rows, cols]");
            }

            int cols = output.Dimensions[1];
            int rows = output.Dimensions[2];

            List<float[]> outputs = new List<float[]>(cols);
            for (int y = 0; y < rows; y++)
            {
                float[] row = new float[cols];
                for (int x = 0; x < cols; x++)
                {
                    row[x] = output[0, x, y];
                }
                outputs.Add(row);
            }

            List<int> classIds = new List<int>();
            List<float> scores = new List<float>();
            List<Rectangle> boxes = new List<Rectangle>();

            for (int i = 0; i < outputs.Count; i++)
            {
                float[] classesScores = outputs[i].Skip(4).ToArray();
                float maxScore = classesScores.Max();

                int classId = Array.IndexOf(classesScores, maxScore);

                float x = outputs[i][0];
                float y = outputs[i][1];
                float w = outputs[i][2];
                float h = outputs[i][3];

                x -= w / 2;
                y -= h / 2;

                x *= (preprocessed.OriginalWidth + preprocessed.XPadding) / (float)preprocessed.Image.Width;
                y *= (preprocessed.OriginalHeight + preprocessed.YPadding) / (float)preprocessed.Image.Height;
                w *= (preprocessed.OriginalWidth + preprocessed.XPadding) / (float)preprocessed.Image.Width;
                h *= (preprocessed.OriginalHeight + preprocessed.YPadding) / (float)preprocessed.Image.Height;

                x = Math.Max(0, Math.Min(x, preprocessed.OriginalWidth));
                y = Math.Max(0, Math.Min(y, preprocessed.OriginalHeight));
                w = Math.Min(w, preprocessed.OriginalWidth - x);
                h = Math.Min(h, preprocessed.OriginalHeight - y);

                classIds.Add(classId);
                scores.Add(maxScore);
                boxes.Add(new Rectangle((int)x, (int)y, (int)w, (int)h));
            }

            var indices = DnnInvoke.NMSBoxes(
                boxes.Select(b => new Rectangle(b.X, b.Y, b.Width, b.Height)).ToArray(),
                scores.ToArray(),
                0.15f,
                0.30f
            );

            List<Detection> detections = new List<Detection>();
            foreach (var i in indices)
            {
                var box = boxes[i];
                float score = scores[i];
                int classId = classIds[i];

                detections.Add(new Detection
                {
                    Class = _labels[classId],
                    Score = score,
                    Box = new int[] { box.X, box.Y, box.Width, box.Height }
                });
            }

            return detections;
        }
    }
}
