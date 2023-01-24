This is the code of the paper **Emotional Voice Conversion using Multitask Learning with Text-to-speech**, ICASSP 2020 [[link]](https://arxiv.org/abs/1911.06149)



## Prerequisite

Install required packages 

```shell
pip3 install -r requirements.txt
```



## Inference

Few samples and pretraiend model for VC are provided, so you can try with below command.

Samples contain 20 types of sentences and 7 emotions, 140 utterances in total.

[[model download]](http://gofile.me/4B76q/yobaWLDtb)

[[samples download]](http://gofile.me/4B76q/RkQxQuKvY)

```shell
python3 generate.py --init_from <model_path> --gpu <gpu_id> --out_dir <out_dir>
```

Below is an example of generated wav.

It means the model takes contents of `(fear, 20th contents)` and style of `(anger, 2nd contents)` to make `(anger, 20th contents)`.

```shell
pretrained_model_fea_00020_ang_00002_ang_00020_input_mel.wav
```


## Training

You can train your own dataset, by changing contents of `dataset.py`

```shell
# remove silence within wav files
python3 trimmer.py --in_dir <in_dir> --out_dir <out_dir>

# Extract mel/lin spectrogram and dictionary of characters/phonemes
python3 preprocess.py --txt_dir <txt_dir> --wav_dir <wav_dir> --bin_dir <bin_dir>

# train the model, --use_txt will control vc path or tts path
python3 main.py -m <message> -g <gpu_id> --use_txt <0~1, higher value means y_t batch is more sampled>
```



1 - 

target : In this article, we have shown that "Voice Conversion"
based on feeling
Uses. Although research 2 that uses the "multipurpose training" method
There is a lot about "audio conversion", the conversion function
Voices, due to the lack of preservation of linguistic information, from the multi-purpose method
used educationally. By combining these two methods, the previous problem is solved
will be. A single model for both "sound conversion" and
"Speech to text" is optimized and the created system will cause
It can be used for both methods.

2 - 

The result shows that "multipurpose training" to a significant extent
It reduces the amount of error of the word VGR and the evaluations as well
it shows.
The content decoder will also increase the quality
"Sound recognition and reconstruction" will be.
And the hardware platform related to VC and TTS to its minimum extent
will receive

3 - ketabkhooneh ha

//import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util//
