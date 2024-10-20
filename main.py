import click
import torch
import torchaudio
from streaming_chattts import StreamingChatTTS

torch.manual_seed(6666)


@click.command()
@click.argument("text")
def main(text: str):
    chatts = StreamingChatTTS()
    audio = torch.cat(list(chatts.generate(text)), dim=1).cpu()
    torchaudio.save("output.wav", audio, 24000)

if __name__ == "__main__":
    main()
