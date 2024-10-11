import time

import torch
import torchaudio
from streaming_dvae import StreamingDVAE

from gpt import GPT


torch.manual_seed(6666)
sdvae = StreamingDVAE()
gpt = GPT(
    {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 20,
        "use_cache": False,
        "max_position_embeddings": 4096,
        "spk_emb_dim": 192,
        "spk_KL": False,
        "num_audio_tokens": 626,
        "num_text_tokens": 21178,
        "num_vq": 4,
    },
    "asset/Embed.safetensors",
    "asset/gpt",
    "asset/tokenizer",
    device="cpu",
)

text = "This repo contains the code for annotating English Fisher Speech Part I and II Transcripts"
audios = []
is_first = True

begin = time.time()
for tokens in gpt.generate(text):
    for audio in sdvae.streaming_decode(tokens):
        if is_first:
            is_first = False
            print(f"{round(time.time() - begin, 3)}s")
        audios.append(audio)
audios.append(sdvae.decode_caches())
audios = torch.cat(audios, dim=1)
print(audios.shape)
torchaudio.save("output.wav", audios.cpu(), 24000)
