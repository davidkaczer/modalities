import re

import imageio.v3 as iio
import torch
import torchaudio


def video_loader(key, data):
    """Based on the torch_video decoder in webdataset
    https://github.com/webdataset/webdataset/blob/5b12e0ba78bfb64741add2533c5d1e4cf088ffff/webdataset/autodecode.py#L394
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    frames = iio.imread(data, index=None, format_hint=key)
    frames = torch.tensor(frames)

    ## Note: this does not exactly match the audio returned by torchvision.io.read_video
    audio = torchaudio.load(data)[0]

    return (frames, audio)


def audio_loader(key, data):
    """Based on the torch_audio decoder in webdataset
    https://github.com/webdataset/webdataset/blob/5b12e0ba78bfb64741add2533c5d1e4cf088ffff/webdataset/autodecode.py#L418
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    return torchaudio.load(data)
