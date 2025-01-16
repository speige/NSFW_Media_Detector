using Emgu.CV.Linemod;
using NSFW_Media_Detector.Image;
using NSFW_Media_Detector.Video;
using System.Diagnostics;

namespace NSFW_Media_Detector.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var input = args[0];

            //NOTE: Sometimes while debugging there will be a memory exception due to reading unmanaged memory that has been moved. Pinning/etc wasn't helpful. Hoping that running in release mode without debugger attached will fix it. If not, may need to use a separate process for the analysis so it can be restarted after crashes.
            if (input.EndsWith(".jpg", StringComparison.InvariantCultureIgnoreCase))
            {
                using (var detector = new NSFWDetector_Ensemble())
                {
                    var bytes = File.ReadAllBytes(input);
                    var throwAway = detector.CalcNSFWProbability(bytes); // 1st run has slow startup, it will throw off performance Stopwatch

                    var stopWatch = new Stopwatch();
                    stopWatch.Start();
                    var result = detector.CalcNSFWProbability(bytes);
                    stopWatch.Stop();
                    Console.WriteLine("NSFW Probability: " + result);
                    Console.WriteLine("Milliseconds to detect: " + TimeSpan.FromMilliseconds(stopWatch.ElapsedMilliseconds).TotalSeconds);
                }
            }
            else if (input.EndsWith(".mp4", StringComparison.InvariantCultureIgnoreCase))
            {
                using (var detector = new NSFWDetector_Ensemble())
                {
                    var outputPath = Path.Combine(Path.GetDirectoryName(input), Path.GetFileNameWithoutExtension(input), "frames");
                    Directory.CreateDirectory(outputPath);

                    if (Directory.GetFiles(outputPath, "*.*", SearchOption.AllDirectories).Length > 0)
                    {
                        throw new Exception("Directory not empty");
                    }

                    //todo: add multi-threading
                    int counter = 1;
                    VideoUtils.ExtractFrames(input, bytes =>
                    {
                        try
                        {
                            var probability = (int)(detector.CalcNSFWProbability(bytes) * 100);
                            File.WriteAllBytes(Path.Combine(outputPath, $"{counter}_{probability}.jpg"), bytes);
                        }
                        catch (Exception e)
                        {
                            Console.Write(e.Message);
                        }

                        counter++;
                    });
                }
            }
            else
            {
                throw new Exception("unsupported file format");
            }
        }
    }
}