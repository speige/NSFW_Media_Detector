using FFMpegCore;

namespace NSFW_Media_Detector.Video
{
    public class VideoUtils
    {
        static VideoUtils()
        {
            GlobalFFOptions.Configure(options =>
            {
                options.BinaryFolder = Path.Combine(Path.GetDirectoryName(Environment.ProcessPath), "ffmpeg_binaries");
            });
        }

        public static void ExtractFrames(string inputVideo, Action<byte[]> action)
        {
            var result = FFMpegArguments
                      .FromFileInput(inputVideo)
                      .OutputToPipe(new MultiImagePipeSink(action), options => options
                        .ForceFormat("image2pipe")
                        .WithVideoCodec("mjpeg")
                        .WithCustomArgument("-vsync 0")
                        .WithCustomArgument("-q:v 1")
                      )
                      .ProcessSynchronously();
        }
    }
}