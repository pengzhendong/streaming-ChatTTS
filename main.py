import time

import ChatTTS
import torch
import torchaudio

from streaming_dvae import StreamingDVAE


torch.manual_seed(6666)


class RefineTextParams:
    prompt = ""
    top_P = 0.7
    top_K = 20
    temperature = 0.7
    repetition_penalty = 1.0
    max_new_token = 384
    min_new_token = 0
    show_tqdm = True
    ensure_non_empty = True
    manual_seed = None


chat = ChatTTS.Chat()
chat.load(compile=True)
class InferCodeParams(RefineTextParams):
    prompt: str = "[speed_5]"
    spk_emb = chat.sample_random_speaker()
    spk_smp = None
    txt_smp = None
    temperature = 0.1
    repetition_penalty = 1.05
    max_new_token = 2048
    stream_batch = 4
    stream_speed = 12000
    pass_first_n_batches = 2

sdvae = StreamingDVAE()

text = "This repo contains the code for annotating English Fisher Speech Part I and II Transcripts"
begin = time.time()
# text_tokens = chat._refine_text(text, "cpu", RefineTextParams()).ids
# text_tokens = [i[i.less(chat.tokenizer.break_0_ids)] for i in text_tokens]
# text = chat.tokenizer.decode(text_tokens)

audios = []
is_first = True
num_frames = 0
for result in chat._infer_code(text, True, "cpu", True, InferCodeParams()):
    ids = result.ids[0].T[None, :, num_frames:]
    num_frames += ids.shape[2]
    for audio in sdvae.streaming_decode(ids):
        if is_first:
            is_first = False
            print(f"{round(time.time() - begin, 3)}s, {num_frames}")
        audios.append(audio)
audios.append(sdvae.decode_caches())
audios = torch.cat(audios, dim=1)
torchaudio.save("output.wav", audios.cpu(), 24000)
