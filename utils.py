import torch
import torch.nn.functional as F
from preprocessing.preprocessing_utils import *
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from preprocessing import k400, nyu_d, audioset
import scipy
import torchaudio
from torchvision import transforms
import os

def cosine_similarity_loss(logits):
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    
    loss_s = F.cross_entropy(logits.T, labels)
    loss_t = F.cross_entropy(logits, labels)

    loss = (loss_s + loss_t) / 2.0
    return loss



def load_n_transform_image_data(img_paths, device):

    image_transform = get_img_preprocess_with_rgb()

    data = []

    for img_path in img_paths:
        with open(img_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = image_transform(image).to(device)   

        data.append(image)
    
    return torch.stack(data, dim=0)


def load_n_transform_video_data(vid_paths, device, clip_duration=2, clips_per_video=5, sample_rate=16000):

    clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )

    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
    video_transform = get_vid_preprocess()

    for idx in range(len(vid_paths)):
        video_path = vid_paths[idx]

        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = k400.get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip).to(device) for clip in all_video]
        all_video = k400.SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)

    return all_video

def load_n_tranform_depth_data(depth_paths, device):
    
    min_depth=0.01
    max_depth=75.0
    depth_transform = get_depth_preprocess()

    sensor_to_params = {
        "kv1": {
            "baseline": 0.075,
        },
        "kv1_b": {
            "baseline": 0.075,
        },
        "kv2": {
            "baseline": 0.075,
        },
        "realsense": {
            "baseline": 0.095,
        },
        "xtion": {
            "baseline": 0.095, # guessed based on length of 18cm for ASUS xtion v1
        },
    }

    all_disparity = []
    for depth_path in depth_paths:
        
        
        depth_file = os.path.join(depth_path, os.listdir(depth_path)[0])
        intrinsics_file = os.path.join('/'.join(depth_path.split('/')[:-1]), 'intrinsics.txt')
        sensor_type = depth_path.split('/')[8]
        with open(intrinsics_file, 'r') as fh:
            lines = fh.readlines()
            focal_length = float(lines[0].strip().split()[0])
        baseline = sensor_to_params[sensor_type]["baseline"]
        depth_image = np.array(Image.open(depth_file))
        depth = np.array(depth_image).astype(np.float32)
        depth_in_meters = depth / 1000. 
        if min_depth is not None:
            depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
        disparity = baseline * focal_length / depth_in_meters
        disparity = disparity / max_depth
        
        disparity = torch.from_numpy(disparity).float().unsqueeze(0) 

        disparity = depth_transform(disparity).to(device)
        all_disparity.append(disparity)
    
    return torch.stack(all_disparity, dim=0)


def load_n_transform_depth_nyu_d_data(data_file_path, split_file_path, img_idxs, device):

    train_test_split = scipy.io.loadmat(split_file_path)
    test_idx = [idx.item() - 1 for idx in train_test_split['testNdxs']]

    depth, image, gt_list = nyu_d.get_data(data_file_path, test_idx)

    depth_transform = get_depth_preprocess()

    all_depth = []
    for idx in img_idxs:

        disparities = nyu_d.depth_to_disparity(depth)

        disparity_map = np.transpose(disparities[idx, :, :], (1, 0))
        disparity = torch.tensor(disparity_map)
        disparity = disparity.clamp(min=0.01)

        disparity = disparity.clamp(max=255)
        disparity /= 255
        disparity = disparity.unsqueeze(0) 

        disparity = depth_transform(disparity).to(device)

        all_depth.append(disparity)

    return torch.stack(all_depth, dim=0)

def load_n_transform_text_data(texts, bpe_path, device):
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in texts]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_n_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    print(mean,std)
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = audioset.get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = audioset.waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)