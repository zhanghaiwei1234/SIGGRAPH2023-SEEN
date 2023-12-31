# SIGGRAPH2023-SEEN

# In the Blink of an Eye: Event-based Emotion Recognition
Haiwei Zhang, [Jiqing Zhang](https://zhangjiqing.com), [Bo Dong](https://dongshuhao.github.io/), Pieter peers, Wenwei Wu, Xiaopeng Wei, Felix Heide, [Xin Yang](https://xinyangdut.github.io/)

[[paper]( https://doi.org/10.1145/3588432.3591511)] [[dataset](http://www.dluticcd.com/)]

<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/introduce.png"></a>
Demonstration of a wearable single-eye emotion recognition prototype system consisting with a bio-inspired event-based camera (DAVIS346) and a low-power NVIDIA Jetson TX2 computing device. Event-based cameras simultaneously provide intensity and corresponding events, which we input to a newly designed lightweight Spiking Eye Emotion Network (SEEN) to effectively extract and combine spatial and temporal cues for emotion recognition. Given a sequence, SEEN takes the start and end intensity frames (green boxes) along with $n$ intermediate event frames (red boxes) as input. Our prototype system consistently recognizes emotions based on single-eye areas under different lighting conditions at $30$ FPS. 

## Abstract
We introduce a wearable single-eye emotion recognition device and a real-time approach to recognizing emotions from partial observations of an emotion that is robust to changes in lighting conditions. At the heart of our method is a bio-inspired event-based camera setup and a newly designed lightweight Spiking Eye Emotion Network (SEEN). Compared to conventional cameras, event-based cameras offer a higher dynamic range (up to 140 dB vs. 80 dB) and a higher temporal resolution (in the order of $\mu$s vs. 10s of $m$s). Thus, the captured events can encode rich temporal cues under challenging lighting conditions. However, these events lack texture information, posing problems in decoding temporal information effectively. SEEN tackles this issue from two different perspectives. First, we adopt convolutional spiking layers to take advantage of the spiking neural network's ability to decode pertinent temporal information. Second, SEEN learns to extract essential spatial cues from corresponding intensity frames and leverages a novel weight-copy scheme to convey spatial attention to the convolutional spiking layers during training and inference. We extensively validate and demonstrate the effectiveness of our approach on a specially collected Single-eye Event-based Emotion (SEE) dataset. To the best of our knowledge, our method is the first eye-based emotion recognition method that leverages event-based cameras and spiking neural networks.

## our dataset SEE

To address this lack of training data for event-based emotion recognition, we collect a new Single-eye Event-based Emotion (SEE) dataset;  SEE contains data from 111 volunteers captured with a DAVIS346 event-based camera placed in front of the right eye and mounted on a helmet; SEE contains videos of 7 emotions under four different lighting conditions: normal, overexposure, low-light, and high dynamic range (HDR) (Figure 3(a)). The average video length ranges from 18 to 131 frames, with a mean frame number of 53.5 and a standard deviation of 15.2 frames, reflecting the differences in the duration of emotions between subjects. In total, SEE contains 2, 405/128, 712 sequences/frames with corresponding raw events for a total length of 71.5 minutes (Figure 3(b)), which we split in 1, 638 and 767 sequences for training and testing, respectively.
<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/dataset.png"></a>

our dataset can be found [here](http://www.dluticcd.com/).

## our Neworks(SEEN)
<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/pipeline-v5.png"></a>


## Requirements

* Python == 3.7
 
* Pytorch == '1.9.0'

* CUDA == 11.4

* torchvision==0.9.0

## Training

* Step 1: download SEE dataset.
<!-- * Step 2: Specifying the command line -->
* Step 2: run main.py
```bash
CUDA_VISIBLE_DEVICES=4  python main.py --root_path /video-emotion-classfication/dataset-60  --event_video_path event_30 --frame_video_path frame  --annotation_path emotion_new_adjust2.json --result_path  sometest/test_163   --dataset emotion --n_classes 7 --batch_size 32 --n_threads 16 --checkpoint 100 --inference --no_val --tensorboard --weight_decay 1e-3 --n_epochs 180 --sample_size 90 --no_hflip --sample_duration 4  --inference_batch_size 120 --inference_stride 0  --sample_t_stride 4  --inference_sample_duration 4 --thresh 0.3 --lens 0.5 --decay 0.2 --beta 0 --learning_rate 0.015 --lr_scheduler singlestep
```
<!-- 

## Documents
* More [Usages](moco-doc/usage.md)
* Detailed [HTTP APIs](moco-doc/apis.md) or [Socket APIs](moco-doc/socket-apis.md)
* Detailed [REST API](moco-doc/rest-apis.md)
* Detailed [Websocket API](moco-doc/websocket-apis.md)
* [Global Settings](moco-doc/global-settings.md) for multiple configuration files.
* [Command Line Usages](moco-doc/cmd.md)
* [Extend Moco](moco-doc/extending.md) if current API does not meet your requirement. -->

