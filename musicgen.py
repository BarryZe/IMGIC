from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def gen_music(descriptions, filename):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=10)  # generate 10 seconds.

    wav = model.generate([descriptions])  # generates a sample.

    for idx, one_wav in enumerate(wav):
        # Save file, with loudness normalization at -14 db LUFS.
        name = filename[14:].split('.')[0]
        audio_write(f'static/output/{name}', one_wav.cpu(), model.sample_rate, strategy="loudness")
