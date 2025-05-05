using Emgu.CV.Dnn;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace NSFW_Media_Detector.Image.Models.NudeNet
{
    public class BaseCleanScreenDetector : BaseNSFWImageDetector<List<BaseCleanScreenDetector.Detection>>
    {
        protected const float confidenceThreshold = .6f;

        public enum Label
        {
            ANUS_NUDE = 0,
            ARMPITS = 1,
            BELLY_CLOTHED = 2,
            BELLY_NUDE = 3,
            BREAST_CLOTHED = 4,
            BUTTOCKS_CLOTHED = 5,
            BUTTOCKS_IMMODEST = 6,
            BUTTOCKS_NUDE = 7,
            CROTCH_CLOTHED = 8,
            CROTCH_IMMODEST = 9,
            FACE = 10,
            FEET = 11,
            FEMALE_BREAST_IMMODEST = 12,
            FEMALE_BREAST_NUDE = 13,
            FEMALE_CROTCH_NUDE = 14,
            HAND = 15,
            MALE_BREAST_NUDE = 16,
            MALE_CROTCH_NUDE = 17,
            PERSON = 18
        }

        protected readonly Dictionary<Label, float> _labelWeights;
        protected readonly float _maxLabelWeight;

        public BaseCleanScreenDetector()
              : base(Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), @"Models\CleanScreen\cleanscreen_480n.onnx"), "images", "output0", 480, 480)
        {
            _shape = new int[] { 1, 3, _resizeHeight, _resizeWidth };
            _pixelTransformer = x => x / 255;
            _padToMaintainAspectRatio = true;
            _tensorLayoutChannelFirst = true;

            var labelsWithWeights = new (Label label, float weight)[]
            {
                (Label.ANUS_NUDE, 1.0f),
                (Label.ARMPITS, 0f),
                (Label.BELLY_CLOTHED, 0f),
                (Label.BELLY_NUDE, .1f),
                (Label.BREAST_CLOTHED, 0f),
                (Label.BUTTOCKS_CLOTHED, 0f),
                (Label.BUTTOCKS_IMMODEST, .25f),
                (Label.BUTTOCKS_NUDE, 1.0f),
                (Label.CROTCH_CLOTHED, 0f),
                (Label.CROTCH_IMMODEST, .25f),
                (Label.FACE, 0f),
                (Label.FEET, 0f),
                (Label.FEMALE_BREAST_IMMODEST, .25f),
                (Label.FEMALE_BREAST_NUDE, 1.0f),
                (Label.FEMALE_CROTCH_NUDE, 1.0f),
                (Label.HAND, 0f),
                (Label.MALE_BREAST_NUDE, .05f),
                (Label.MALE_CROTCH_NUDE, 1.0f),
                (Label.PERSON, 0f),
            };

            _labelWeights = labelsWithWeights.ToDictionary(x => x.label, x => x.weight);
            _maxLabelWeight = _labelWeights.Values.Max();
        }

        public class Detection
        {
            public Label Label { get; set; }
            public float Score { get; set; }
            public Rectangle Box { get; set; }
        }

        protected override float DetectionToProbability(List<Detection> detection)
        {
            return (float)Math.Clamp(Math.Round(detection.Select(x => _labelWeights[x.Label]).Sum(), 2), 0, 1);
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

            var widthScale = ((float)preprocessed.OriginalWidth) / (_resizeWidth - preprocessed.XPadding - preprocessed.XPadding);
            var heightScale = ((float)preprocessed.OriginalHeight) / (_resizeHeight - preprocessed.YPadding - preprocessed.YPadding);

            for (int i = 0; i < outputs.Count; i++)
            {
                float[] classesScores = outputs[i].Skip(4).ToArray();
                float maxScore = classesScores.Max();

                if (maxScore < confidenceThreshold)
                {
                    continue;
                }

                int classId = Array.IndexOf(classesScores, maxScore);

                float centerX = outputs[i][0];
                float centerY = outputs[i][1];
                float halfWidth = outputs[i][2] / 2;
                float halfHeight = outputs[i][3] / 2;

                var x1 = (centerX - halfWidth - preprocessed.XPadding) * widthScale;
                var y1 = (centerY - halfHeight - preprocessed.YPadding) * heightScale;
                var x2 = (centerX + halfWidth - preprocessed.XPadding) * widthScale;
                var y2 = (centerY + halfHeight - preprocessed.YPadding) * heightScale;

                x1 = Math.Max(0, Math.Min(x1, preprocessed.OriginalWidth));
                y1 = Math.Max(0, Math.Min(y1, preprocessed.OriginalHeight));
                x2 = Math.Max(0, Math.Min(x2, preprocessed.OriginalWidth));
                y2 = Math.Max(0, Math.Min(y2, preprocessed.OriginalHeight));

                classIds.Add(classId);
                scores.Add(maxScore);
                boxes.Add(new Rectangle((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)));
            }

            var indices = DnnInvoke.NMSBoxes(
                boxes.Select(b => new System.Drawing.Rectangle(b.X, b.Y, b.Width, b.Height)).ToArray(),
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
                    Label = (Label)classId,
                    Score = score,
                    Box = box
                });
            }

            return detections;
        }
    }
}
