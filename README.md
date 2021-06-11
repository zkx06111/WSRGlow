# WSRGlow

The official implementation of the Interspeech 2021 paper *WSRGlow: A Glow-based Waveform Generative Model for Audio Super-Resolution*. Audio samples can be found [here](https://zkx06111.github.io/wsrglow/).

The configs for model architecture and training scheme is saved in `config.yaml`. You can overwrite some of the attributes by adding the `--hparams` flag when running a command. The general way to run a python script is

`python $SRC$ --config $CONFIG$ --hparams $KEY1$=$VALUE1$,$KEY2$=$VALUE2$,...`

See `hparams.py` for more details.

### To prepare data

Before training, you need to binarize the data first. The raw wav files should be put in the `hparams['raw_data_path']`. The binarized data would be put in the `hparams['binary_data_path']`.

The command to binarize is

`python binarizer.py --config config.yaml`

### To modify the architecture of the model

The current WSRGlow model in `model.py` is designed for x4 super-resolution and takes waveform, spectrogram and phase information as input.

### To train

Run `python train.py --config config.yaml` on a GPU.

### To infer

Run `python infer.py --config config.yaml` on a GPU, modify the code for the correct path of checkpoints and wav files.
