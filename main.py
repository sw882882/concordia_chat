from openai import OpenAI
from whispercpp import Whisper
import time
import ffmpeg
import numpy as np
from pydub import AudioSegment
import os
import speech_recognition as sr

ASSISTANT_ID = "asst_cFJlHpJu7ly1ooYrL3v2Q2AU"

client = OpenAI()


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(ASSISTANT_ID, thread, user_input)
    return thread, run


def continue_thread_and_run(thread, user_input):
    run = submit_message(ASSISTANT_ID, thread, user_input)
    return run


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
        os.system(f'say -v "Samantha" "{m.content[0].text.value}"')


def local_cpu_transcribe(audio_file):
    # TODO for deployment either add download code here or manually do it
    try:
        y, _ = (
            ffmpeg.input(audio_file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=32768.0)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

    return Whisper.from_pretrained("base.en").transcribe(arr)


def milliseconds_until_sound(sound, silence_threshold_in_decibels=-30.0, chunk_size=10):
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[
        trim_ms : trim_ms + chunk_size
    ].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def openai_transcribe_audio(filepath):
    audio = AudioSegment.from_file(filepath)
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]
    # save
    trimmed.export(f"/tmp/trimmed.wav", format="wav")
    with open("/tmp/trimmed.wav", "rb") as audio_data:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        return transcription.text


r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)

    audio = r.listen(source, timeout=5)

with open("temp.wav", "wb") as file:
    file.write(audio.frame_data)
# works but quite bad and slow
# message = local_cpu_transcribe("./test3.m4a")
message = openai_transcribe_audio("./temp.wav")
print(message)

thread1, run1 = create_thread_and_run(message)
run1 = wait_on_run(run1, thread1)
pretty_print(get_response(thread1))
# bash command
### multi response conversation example
# message = "What organizations is the school approved by?"
# thread1, run1 = create_thread_and_run(message)
# run1 = wait_on_run(run1, thread1)
# pretty_print(get_response(thread1))

# message = "Can you explain to me what they do"
# run1 = continue_thread_and_run(thread1, message)
# run1 = wait_on_run(run1, thread1)
# pretty_print(get_response(thread1))

# TODO add direction documents
# TODO Handle language
