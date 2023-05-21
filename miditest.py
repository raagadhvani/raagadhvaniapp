from midi2audio import FluidSynth
# using the default sound font in 44100 Hz sample rate
fs = FluidSynth()
fs.midi_to_audio('F:/Meghana/raagadhvani/sample.mid', 'output.wav')

