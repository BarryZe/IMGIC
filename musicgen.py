from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def gen_music(descriptions):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=10)  # generate 10 seconds.

    wav = model.generate([descriptions])  # generates 2 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
<<<<<<< HEAD
        audio_write(f'static/output/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
=======
        audio_write(f'static/output/{descriptions}', one_wav.cpu(), model.sample_rate, strategy="loudness")
>>>>>>> 934e5f7474a486031eebd3c45c55808e0865ada4
