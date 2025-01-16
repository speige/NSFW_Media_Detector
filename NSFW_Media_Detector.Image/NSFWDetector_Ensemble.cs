using NSFW_Media_Detector.Image.Models.NudeNet;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace NSFW_Media_Detector.Image
{
    public class NSFWDetector_Ensemble : IDisposable
    {
        Dictionary<INSFWImageDetector, float> _weightPerDetector;
        public NSFWDetector_Ensemble()
        {
            _weightPerDetector = new Dictionary<INSFWImageDetector, float>()
            {
                { new NudeNetDetector(), 1.0f },
            };

            var total = _weightPerDetector.Select(x => x.Value).Sum();
            _weightPerDetector = _weightPerDetector.ToDictionary(x => x.Key, x => x.Value / total);
        }

        public float CalcNSFWProbability(byte[] imageBytes)
        {
            using (var memoryStream = new MemoryStream(imageBytes))
            using (var image = SixLabors.ImageSharp.Image.Load<Rgba32>(memoryStream))
            {
                // .AsParallel() could be useful if we have a lot of detectors to ensemble
                var result = _weightPerDetector.Select(x =>
                {
                    var score = Math.Clamp(x.Key.CalcNSFWProbability(image) * 200 - 100, -100, 100);
                    return new { score, weight = x.Value };
                }).ToList();

                var weightedAverage = result.Select(x => x.score * x.weight).Sum();

                return (float)Math.Clamp((weightedAverage + 100) / 200, 0, 1);
            }
        }

        public void Dispose()
        {
            foreach (var detector in _weightPerDetector.Keys)
            {
                detector.Dispose();
            }
        }
    }
}
