using FFMpegCore.Pipes;

namespace NSFW_Media_Detector.Video
{
    public class MultiImagePipeSink : IPipeSink
    {
        private readonly Action<byte[]> _onFrame;
        private readonly List<byte> _frameBuffer = new List<byte>();

        public MultiImagePipeSink(Action<byte[]> onFrame)
        {
            _onFrame = onFrame;
        }

        public string GetFormat() => "image2pipe";

        public async Task ReadAsync(Stream inputStream, CancellationToken cancellationToken)
        {
            var buffer = new byte[8192];
            int bytesRead;

            while ((bytesRead = await inputStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
            {
                for (int i = 0; i < bytesRead; i++)
                {
                    _frameBuffer.Add(buffer[i]);

                    int count = _frameBuffer.Count;
                    if (count >= 2 &&
                        _frameBuffer[count - 2] == 0xFF &&
                        _frameBuffer[count - 1] == 0xD9)
                    {
                        _onFrame(_frameBuffer.ToArray());

                        _frameBuffer.Clear();
                    }
                }
            }

            _onFrame(_frameBuffer.ToArray());
        }
    }
}
