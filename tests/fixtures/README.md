# Test fixtures

`jfk.wav`: whisper.cpp's JFK sample ("And so my fellow Americans, ask not
what your country can do for you..."), 16 kHz mono. The recording is a
U.S. government work in the public domain.

`tone.flac`: 0.1 s, mono 16 kHz, 16-bit 440 Hz sine (amplitude 10000),
generated with Python's wave/struct/math modules and encoded using
`flac --no-padding --no-seektable`. Used without a model or subprocess in tests.
