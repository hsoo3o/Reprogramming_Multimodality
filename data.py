
from torch.utils.data import Dataset
import os
import numpy as np
from preprocessing.preprocessing_utils import *
from preprocessing import sun_rgb_d, nyu_d, audioset, esc50
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
import pandas as pd 
from torchvision import transforms
import torchaudio
    
class SunRgbDDataset(Dataset):
    def __init__(self, root, tag):
        super(SunRgbDDataset, self).__init__()

        self.gt_classes = sun_rgb_d.get_gt(root)

        self.depth_path, self.image_path, self.target_list = sun_rgb_d.get_data(root, self.gt_classes, tag)
        
        self.image_transform = get_img_preprocess_with_rgb()
        self.depth_transform = get_depth_preprocess()
    
    def __len__(self):
        return len(self.depth_path)

    def __getitem__(self, idx):
        
        image_file_path =  os.path.join(self.image_path[idx], os.listdir(self.image_path[idx])[0])
        
        with open(image_file_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = self.image_transform(image)   

        disparity = sun_rgb_d.convert_depth_to_disparity(self.depth_path[idx])
        disparity = disparity.unsqueeze(0) 

        disparity = self.depth_transform(disparity)

        return image, disparity


    
class NyuDDataset(Dataset):
    def __init__(self, data_file_path, data_idx):

        self.depth, self.image, self.gt_list = nyu_d.get_data(data_file_path, data_idx)

        self.data_idx = data_idx

        self.image_transform = get_img_preprocess_with_rgb()
        self.depth_transform = get_depth_preprocess()

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        rgb_image = np.transpose(self.image[idx, :, :, :], (2, 1, 0))  
        img = Image.fromarray(rgb_image)        
        image = self.image_transform(img)


        disparities = nyu_d.depth_to_disparity(self.depth)

        disparity_map = np.transpose(disparities[idx, :, :], (1, 0))
        disparity = torch.tensor(disparity_map)
        disparity = disparity.clamp(min=0.01)

        disparity = disparity.clamp(max=255)
        disparity /= 255
        disparity = disparity.unsqueeze(0) 

        disparity = self.depth_transform(disparity)

        return image, disparity

    def depth_to_disparity(self, depth_map, focal_length=582.62, baseline=0.075):

        depth_map[depth_map == 0] = 1e-6
        disparity_map = (focal_length * baseline) / depth_map
        return disparity_map




class AudioSetDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 device,
                 image_process,
                 num_mel_bins=128,
                 target_length=204,
                 sample_rate=16000,
                 clip_duration=2,
                 clips_per_video=3,
                 audio_mean=-4.268,
                 audio_std=9.138,
        ):
        super(AudioSetDataset, self).__init__()

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )

        self.frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=1)



        self.audio_list, self.video_list = audioset.get_audio_n_video_path(root_dir)

        self.image_process = image_process
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.device = device
    

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        audio_path = self.audio_list[idx]
        video_path = self.video_list[idx]

        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            # **{"sample_rate": self.sample_rate},
        )

        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        all_clips_timepoints = audioset.get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )

        all_audio_clips = []
        # all_video_frames = []

        for clip_timepoints in all_clips_timepoints:
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")

            video_frames = clip["video"]

            video_frame = video_frames[:,int(video_frames.shape[1]//2),:]
            video_frame = torch.squeeze(video_frame, dim=1)
            video_frame = video_frame / 255.0  

            to_pil = transforms.ToPILImage()

            frame_image = to_pil(video_frame)

            image = self.image_process(frame_image)  

            # all_video_frames.append(image)

            
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(
                    clip_timepoints[1] * self.sample_rate
                ),
            ]
            waveform_melspec = audioset.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_audio_clips.append(waveform_melspec)

        
        # all_video_frames = torch.stack(all_video_frames, dim=0)


        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        all_audio_clips = [normalize(ac).to(self.device) for ac in all_audio_clips]

        all_audio_clips = torch.stack(all_audio_clips, dim=0)

        # all_video_frames = torch.squeeze(all_video_frames, dim=1)

        return image, all_audio_clips
    


class Esc50Dataset(Dataset):
    def __init__(self, 
                 root_dir,
                 csv_path,
                 device,
                 num_mel_bins=128,
                 target_length=204,
                 sample_rate=16000,
                 clip_duration=2,
                 clips_per_video=3,
                 audio_mean=-4.268,
                 audio_std=9.138,
        ):
        super(Esc50Dataset, self).__init__()

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )


        self.audio_list = esc50.get_audio_path(root_dir, csv_path)

        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.device = device
    

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):

        audio_path = self.audio_list[idx]


        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        all_clips_timepoints = audioset.get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )

        all_audio_clips = []
        # all_video_frames = []

        for clip_timepoints in all_clips_timepoints:
            
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(
                    clip_timepoints[1] * self.sample_rate
                ),
            ]
            waveform_melspec = audioset.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_audio_clips.append(waveform_melspec)

        

        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        all_audio_clips = [normalize(ac).to(self.device) for ac in all_audio_clips]

        all_audio_clips = torch.stack(all_audio_clips, dim=0)

        # all_video_frames = torch.squeeze(all_video_frames, dim=1)

        return all_audio_clips



class AudioSetDataset_video(Dataset):
    def __init__(self, 
                 root_dir,
                 csv_path,
                 device,
                 video_process,
                 num_mel_bins=128,
                 target_length=204,
                 sample_rate=16000,
                 clip_duration=2,
                 clips_per_video=3,
                 audio_mean=-4.268,
                 audio_std=9.138,
        ):
        super(AudioSetDataset_video, self).__init__()

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )

        self.frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)


        self.audio_list, self.video_list = audioset.get_audio_n_video_path(root_dir)

        self.video_process = video_process
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.device = device
    

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        audio_path = self.audio_list[idx]
        video_path = self.video_list[idx]

        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": self.sample_rate},
        )

        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        all_clips_timepoints = audioset.get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )

        all_audio_clips = []
        all_video_clips = []

        for clip_timepoints in all_clips_timepoints:
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = self.frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video_clips.append(video_clip)

            
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(
                    clip_timepoints[1] * self.sample_rate
                ),
            ]
            waveform_melspec = audioset.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_audio_clips.append(waveform_melspec)


        all_video_clips = [self.video_process(clip) for clip in all_video_clips]
        all_video_clips = audioset.SpatialCrop(224, num_crops=3)(all_video_clips)

        all_video_clips = torch.stack(all_video_clips, dim=0)


        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        all_audio_clips = [normalize(ac).to(self.device) for ac in all_audio_clips]

        all_audio_clips = torch.stack(all_audio_clips, dim=0)

        return all_video_clips, all_audio_clips

class AudioSetDataset_audio(Dataset):
    def __init__(self, 
                 root_dir,
                 device,
                 num_mel_bins=128,
                 target_length=204,
                 sample_rate=16000,
                 clip_duration=2,
                 clips_per_video=3,
                 audio_mean=-4.268,
                 audio_std=9.138,
        ):
        super(AudioSetDataset_audio, self).__init__()

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )


        self.audio_list, self.video_list = audioset.get_audio_n_video_path(root_dir)

        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.device = device
    

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        audio_path = self.audio_list[idx]

        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        all_clips_timepoints = audioset.get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )

        all_audio_clips = []

        for clip_timepoints in all_clips_timepoints:

            
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(
                    clip_timepoints[1] * self.sample_rate
                ),
            ]
            waveform_melspec = audioset.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_audio_clips.append(waveform_melspec)

        


        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        all_audio_clips = [normalize(ac).to(self.device) for ac in all_audio_clips]

        all_audio_clips = torch.stack(all_audio_clips, dim=0)


        return all_audio_clips
    