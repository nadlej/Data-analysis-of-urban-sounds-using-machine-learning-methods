import muda
import pandas as pd

def deform_audio(jams_audio, folder_number, file_name):
    for deform_index in range(len(time_stretching)):

        # Zainicjalizowanie wszystkich deformacji
        pitch_audio_first = muda.deformers.PitchShift(n_semitones=pitch_shifting_first[deform_index])
        pitch_audio_second = muda.deformers.PitchShift(n_semitones=pitch_shifting_second[deform_index])
        time_stretch_audio = muda.deformers.TimeStretch(rate=time_stretching[deform_index])
        drc_audio = muda.deformers.DynamicRangeCompression(preset=DRC[deform_index])
        background_noises_audio = muda.deformers.BackgroundNoise(1, files=background_noises[deform_index], weight_min=0.1, weight_max=0.5)
        file_name_without_extended = file_name.replace(".wav", "")

        # Dla kazdej deformacji zostaje zapisany plik audio oraz adnotacja w formacie JAMS we wskazanej lokalizacji
        for i, jam_out in enumerate(pitch_audio_first.transform(jams_audio)):
            muda.save(localization + str(folder_number) + '/pitch1/audio/' + file_name_without_extended + '_pitch1_' + str(deform_index) + ".wav",
                      localization + str(folder_number) + '/pitch1/jams/' + file_name_without_extended + '_pitch1_' + str(deform_index) + ".jams", jam_out)

        for i, jam_out in enumerate(pitch_audio_second.transform(jams_audio)):
            muda.save(localization + str(folder_number) + '/pitch2/audio/' + file_name_without_extended + '_pitch2_' + str(deform_index) + ".wav",
                      localization + str(folder_number) + '/pitch2/jams/' + file_name_without_extended + '_pitch2_' + str(deform_index) + ".jams", jam_out)

        for jam_out in time_stretch_audio.transform(jams_audio):
            muda.save(localization + str(folder_number) + '/stretch/audio/' + file_name_without_extended + '_stretch_' + str(deform_index) + ".wav",
                      localization + str(folder_number) + '/stretch/jams/' + file_name_without_extended + '_stretch_' + str(deform_index) + ".jams", jam_out)

        for jam_out in drc_audio.transform(jams_audio):
            muda.save(localization + str(folder_number) + '/drc/audio/' + file_name_without_extended + '_drc_' + str(deform_index) + ".wav",
                      localization + str(folder_number) + '/drc/jams/' + file_name_without_extended + '_drc_' + str(deform_index) + ".jams", jam_out)

        for jam_out in background_noises_audio.transform(jams_audio):
            muda.save(localization + str(folder_number) + '/bgnoise/audio/' + file_name_without_extended + '_bgnoise_' + str(deform_index) + ".wav",
                      localization + str(folder_number) + '/bgnoise/jams/' + file_name_without_extended + '_bgnoise_' + str(deform_index) + ".jams", jam_out)

# Inicjalizacja parametrów deformacji
time_stretching = [0.81, 0.93, 1.07, 1.23]
pitch_shifting_first = [-2, -1, 1, 2]
pitch_shifting_second = [-3.5, -2.5, 2.5, 3.5]
DRC = ['music standard', 'film standard', 'speech', 'radio']
background_noises = ['150993__saphe__street-scene-1.wav',
                     '173955__saphe__street-scene-3.wav',
                     '207208__jormarp__high-street-of-gandia-valencia-spain.wav',
                     '268903__yonts__city-park-tel-aviv-israel.wav']
localization = '../../OSTATNI SEMESTR/Praca licencjacka/data/fold'
df = pd.read_csv('../../OSTATNI SEMESTR/Praca licencjacka/data/UrbanSound8K.csv')

# Dla kazdego pliku audio ze zbioru dokonywany jest ciag deformacji
for i in range(8732):
    audio_name = df['slice_file_name'][i]
    folder_index = df['fold'][i]
    jams_audio = muda.load_jam_audio(localization + str(folder_index) + '/oryginal/jams/' + audio_name.replace('wav', 'jams'),
                                     localization + str(folder_index) + '/oryginal/audio/' + audio_name)
    deform_audio(jams_audio, folder_index, audio_name)
    print(str(i+1) + "/8732 done")