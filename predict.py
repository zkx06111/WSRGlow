# based on https://colab.research.google.com/drive/1uJ9bcUdK3VUwWYt0aU1C1mXAXp6JzCLh?usp=sharing

import tempfile
from pathlib import Path
import soundfile as sf
import torch
import cog

from infer import set_hparams, load_ckpt, WaveGlowMelHF, hparams, load_wav, run

class Predictor(cog.Predictor):

    def setup(self):
        print("Loading model...")
        set_hparams(config='config.yaml')
        self.model = WaveGlowMelHF(**hparams['waveglow_config']).cuda()
        load_ckpt(self.model, 'model_ckpt_best.pt')
        self.model.eval()

    @cog.input("input", type=Path, help="Low-sample rate input file in .wav format")
    def predict(self, input):
        if input.suffix != ".wav":
            raise ValueError("Input must be a .wav file")

        print("Loading wav file...")
        lr, sr = load_wav(str(input))

        print(f'sampling rate (lr) = {sr}')
        print(f'lr.shape = {lr.shape}', flush=True)

        print("Running prediction...")
        with torch.no_grad():
            pred = run(self.model, lr, sigma=1)
        print(lr.shape, pred.shape)

        out_path = Path(tempfile.mkdtemp()) / "out.wav"

        print(f'sampling rate = {sr * 2}')
        with out_path.open("wb") as f:
            sf.write(f, pred, sr * 2)

        return out_path
