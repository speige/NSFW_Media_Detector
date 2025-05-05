using Emgu.CV.Reg;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace NSFW_Media_Detector.Image.Models.NudeNet
{
    public class CleanScreenDetector : BaseCleanScreenDetector
    {
        public static readonly Dictionary<Label, bool> POTENTIAL_NSFW_INDICATOR = new Dictionary<Label, bool>();

        protected const int MIN_CROP_SIZE = 64;
        protected const float MAX_CROP_PERCENT = 0.75f;
        protected const int MAX_CROP_COUNT = 3;

        public CleanScreenDetector()
        {
            foreach (var labelWeight in _labelWeights)
            {
                POTENTIAL_NSFW_INDICATOR[labelWeight.Key] = labelWeight.Value > 0;
            }

            POTENTIAL_NSFW_INDICATOR[Label.BELLY_NUDE] = true;
            POTENTIAL_NSFW_INDICATOR[Label.MALE_BREAST_NUDE] = true;
        }

        protected Image<Rgba32> CropBitmap(Image<Rgba32> image, Rectangle box)
        {
            return image.Clone(x => x.Crop(box));
        }

        protected (int x, int y) BoxCenter(Rectangle box)
        {
            return (box.X + (box.Width >> 1), box.Y + (box.Height >> 1));
        }

        protected Rectangle PoseBoundingBox(IEnumerable<Detection> pose)
        {
            var coordinates = new int[] {
                pose.Min(d => d.Box.X),
                pose.Min(d => d.Box.Y),
                pose.Max(d => d.Box.X + d.Box.Width),
                pose.Max(d => d.Box.Y + d.Box.Height)

            };

            return new Rectangle(coordinates[0], coordinates[1], coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]);
        }

        protected bool CanCropForRescan(Rectangle box, (int, int) size)
        {
            if (box.Width < MIN_CROP_SIZE || box.Height < MIN_CROP_SIZE)
                return false;

            if (size.Item1 * size.Item2 * MAX_CROP_PERCENT < box.Width * box.Height)
                return false;

            return true;
        }

        protected override List<Detection> ModelOutputToDetections(PreprocessedImage preprocessed, TensorBase modelOutput)
        {
            var originalDetections = base.ModelOutputToDetections(preprocessed, modelOutput);

            var personLabels = originalDetections.Where(d => d.Label == Label.PERSON).ToList();
            var nonPersonLabels = originalDetections.Where(d => d.Label != Label.PERSON).ToList();

            var poses = personLabels.Select(person =>
            {
                var related = nonPersonLabels
                    .Where(n =>
                    {
                        var center = BoxCenter(n.Box);
                        return center.x >= person.Box.X && center.x <= (person.Box.X + person.Box.Width) &&
                               center.y >= person.Box.Y && center.y <= (person.Box.Y + person.Box.Height);
                    }).ToList();
                related.Add(person);
                return new HashSet<Detection>(related);
            }).ToList();

            var bitmapSize = (preprocessed.OriginalWidth, preprocessed.OriginalHeight);

            var validDetections = new HashSet<Detection>(originalDetections.Where(d => d.Score >= confidenceThreshold));
            var usedDetections = new HashSet<Detection>(validDetections.Where(d => d.Label != Label.PERSON));
            usedDetections.UnionWith(poses.SelectMany(p => p.Where(d => usedDetections.Contains(d))).Where(d => d.Label == Label.PERSON));

            for (int cropCount = 1; cropCount <= MAX_CROP_COUNT; cropCount++)
            {
                var possible = poses
                    .Where(p => p.All(d => !usedDetections.Contains(d)))
                    .Select(p => (pose: p, box: PoseBoundingBox(p)))
                    .Where(p => CanCropForRescan(p.box, bitmapSize))
                    .OrderByDescending(p => p.pose.Select(d => d.Label).Distinct().Count(l => POTENTIAL_NSFW_INDICATOR[l]) +
                        Math.Min(_maxLabelWeight, p.pose.Sum(d => _labelWeights[d.Label])))
                    .ToList();

                var rescanPose = possible.FirstOrDefault();
                if (rescanPose.pose == null) break;

                usedDetections.UnionWith(rescanPose.pose);
                var cropped = CropBitmap(preprocessed.OriginalImage, rescanPose.box);
                var newDetections = base.Detect(cropped)
                    .Where(d => d.Label != Label.PERSON)
                    .Select(d =>
                    {
                        var detection = new Detection() { Label = d.Label, Score = d.Score };
                        detection.Box = new Rectangle
                        (
                            rescanPose.box.X + d.Box.X,
                            rescanPose.box.Y + d.Box.Y,
                            rescanPose.box.X + d.Box.Width,
                            rescanPose.box.X + d.Box.Height
                        );
                        return detection;
                    }).ToList();

                foreach (var detection in newDetections)
                {
                    var center = BoxCenter(detection.Box);
                    var matching = poses.Where(p =>
                    {
                        var person = p.FirstOrDefault(d => d.Label == Label.PERSON);
                        if (person == null) return false;
                        return center.x >= person.Box.X && center.x <= (person.Box.X + person.Box.Width) &&
                               center.y >= person.Box.Y && center.y <= (person.Box.Y + person.Box.Height);
                    });

                    foreach (var match in matching)
                        match.Add(detection);
                }

                usedDetections.UnionWith(newDetections);
                validDetections.UnionWith(newDetections);
            }

            return validDetections.ToList();
        }
    }
}
