# Clips for `make stt`

Drop real recordings here (wav, webm, mp3, m4a, ogg, flac), then:

    SARVAM_API_KEY=... make stt

Each clip is transcribed once through `stt/sarvam.py`; the P50/P95/P100 of the round trip
go to `data/excluded_legs.json` and D3 prints them directly under the latency table.

What to record, and why it matters:
  * real speech in the languages you claim to serve (hi / ta / bn / en, and code-switched) --
    the round trip scales with audio length and with how hard the audio is;
  * 5-10 clips minimum. P95 over 3 samples is the max with a fancier name;
  * the same clips every time, so the number is comparable across runs.

This leg is OUTSIDE the 200 ms budget by definition -- t0 is the instant the server holds a
final transcript. Outside the budget is not the same as invisible, which is the whole reason
this directory exists.
