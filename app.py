import tempfile
from pathlib import Path

import gradio as gr
from yue2 import YuE2Pipeline


MODEL_ID = "m-a-p/YuE2-3B"

# ZeroGPU emulates CUDA during startup and attaches a real GPU for decorated calls.
# Keeping the pipeline here also ensures that model weights are downloaded once.
pipe = YuE2Pipeline.from_pretrained(MODEL_ID, device="cuda", progress=False)


def estimate_duration(style, lyrics, planning_mode, seed):
    del style, planning_mode, seed
    # Typical generations take around a minute; longer lyrics need more queue time.
    return min(300, max(120, 120 + len(lyrics or "") // 12))


def validate_inputs(style, lyrics):
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    if not style:
        raise gr.Error("Describe the musical style.")
    if not lyrics:
        raise gr.Error("Enter lyrics with section labels such as [Verse] and [Chorus].")
    if len(style) > 1_000:
        raise gr.Error("Keep the style prompt under 1,000 characters.")
    if len(lyrics) > 12_000:
        raise gr.Error("Keep the lyrics under 12,000 characters.")
    return style, lyrics


def save_result(song, prefix):
    output_dir = Path(tempfile.mkdtemp(prefix=prefix))
    audio_path = output_dir / "song.flac"
    song.save(audio_path)
    score = song.abc or "No symbolic score was produced in direct-generation mode."
    return str(audio_path), score


def generate_song(style, lyrics, planning_mode, seed):
    style, lyrics = validate_inputs(style, lyrics)

    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            cot=planning_mode,
            seed=int(seed),
        )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc

    return save_result(song, "yue2-generation-")


def generate_cover(style, lyrics, abc_score, seed):
    style, lyrics = validate_inputs(style, lyrics)
    abc_score = (abc_score or "").strip()
    if not abc_score:
        raise gr.Error("Paste a melody-only ABC score for the source song.")
    if len(abc_score) > 100_000:
        raise gr.Error("Keep the ABC score under 100,000 characters.")

    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            abc=abc_score,
            cot="melody",
            seed=int(seed),
        )
    except Exception as exc:
        raise gr.Error(f"Cover generation failed: {exc}") from exc

    return save_result(song, "yue2-cover-")


EXAMPLES = [
    [
        "Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths, uplifting",
        """[Verse]\nCity windows turn to gold\nEvery streetlight has a story\nWe are brave and we are bold\nRunning toward the morning glory\n\n[Chorus]\nStay awake, the night is ours\nWe can dance beneath the stars\nHold this moment, hold it tight\nWe are sparks inside the night\n\n[Outro]\nInside the night""",
        "full",
        42,
    ],
    [
        "Acoustic indie folk, intimate male vocal, fingerpicked guitar, gentle strings",
        """[Verse]\nDust is dancing in the doorway\nSummer settles on the road\nI can hear the old trees whisper\nAll the secrets that they know\n\n[Chorus]\nTake me home across the river\nWhere the evening moves so slow\nIf the wind can find its way there\nThen I know that I can go""",
        "melody",
        831001,
    ],
]


with gr.Blocks(title="YuE2-3B Music Generator") as demo:
    gr.Markdown(
        "# YuE2-3B Music Generator\n"
        "Create a song from a style description and structured lyrics with "
        f"[m-a-p/YuE2-3B](https://huggingface.co/{MODEL_ID})."
    )

    with gr.Tabs():
        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=3):
                    style = gr.Textbox(
                        label="Style",
                        placeholder="Genre, mood, vocal style, instruments…",
                        value="Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths",
                    )
                    lyrics = gr.Textbox(
                        label="Lyrics",
                        lines=16,
                        placeholder="[Verse]\nYour lyrics…\n\n[Chorus]\nYour chorus…",
                    )
                    with gr.Row():
                        planning_mode = gr.Radio(
                            choices=[("Melody + chords", "full"), ("Melody only", "melody"), ("No score", "off")],
                            value="full",
                            label="Symbolic planning",
                        )
                        seed = gr.Number(value=42, precision=0, label="Seed")
                    generate = gr.Button("Generate song", variant="primary")
                with gr.Column(scale=2):
                    audio = gr.Audio(label="Generated song", type="filepath")
                    with gr.Accordion("Generated ABC score", open=False):
                        score = gr.Code(language=None, label="Editable score")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[style, lyrics, planning_mode, seed],
                cache_examples=False,
            )

        with gr.Tab("Cover"):
            gr.Markdown(
                "Rearrange a song from its melody score. Transcribe the source recording with "
                "[SheetSage2](https://huggingface.co/m-a-p/SheetSage2), remove chord symbols, "
                "and paste the melody ABC below. Use lyrics whose sections match the recording."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    cover_style = gr.Textbox(
                        label="New style",
                        placeholder="Jazz-funk, warm lead vocal, Rhodes piano, tight drums…",
                    )
                    cover_lyrics = gr.Textbox(
                        label="Matched lyrics",
                        lines=12,
                        placeholder="[Verse]\nLyrics aligned with the source song…",
                    )
                    cover_abc = gr.Textbox(
                        label="Melody ABC (without chord symbols)",
                        lines=12,
                        placeholder="X:1\nT:Source melody\nM:4/4\nL:1/8\nK:C\n…",
                    )
                    cover_seed = gr.Number(value=831001, precision=0, label="Seed")
                    cover_button = gr.Button("Generate cover", variant="primary")
                with gr.Column(scale=2):
                    cover_audio = gr.Audio(label="Generated cover", type="filepath")
                    with gr.Accordion("Used ABC score", open=False):
                        cover_score = gr.Code(language=None, label="Score")
    gr.Markdown(
        "Generation can take a few minutes. One song is processed at a time. "
        "Model weights are licensed **CC BY-NC 4.0**.\n\n"
        "[Twitter / X](https://x.com/realmrfakename)"
    )

    generate.click(
        fn=generate_song,
        inputs=[style, lyrics, planning_mode, seed],
        outputs=[audio, score],
    )
    cover_button.click(
        fn=generate_cover,
        inputs=[cover_style, cover_lyrics, cover_abc, cover_seed],
        outputs=[cover_audio, cover_score],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
