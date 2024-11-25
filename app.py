import streamlit as st
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import librosa

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

# Configuración de la barra lateral
st.sidebar.title("Opciones de visualización")
page = st.sidebar.radio("Seleccione un modelo", ["Autoencoder y U-Net", "Facebook Denoiser"])

if page == "Autoencoder y U-Net":
    # Mantén tu código existente para Autoencoder y U-Net aquí
    st.title("Modelos Autoencoder y U-Net")
    st.write("Visualización de los modelos personalizados.")
    # TODO: Inserta aquí tu código existente para Autoencoder y U-Net.

    #Cargar modelos
autoencoder_model = keras.models.load_model('autoencoder_m.h5')
unet_model = keras.models.load_model('modeloUnet.h5')  

# Parametros consistentes
n_fft = 2048
hop_length = 512 

def get_spectrogram_features_noisy(file_data, sample_rate=22050, duration_s=1):
    
    #Duracion del sample
    duration = int(sample_rate * duration_s)

    #Cargar audio del archivo
    data, _ = librosa.load(file_data, sr=sample_rate)

    # Pad/Cut
    if len(data) < sample_rate:
        max_offset = np.abs(len(data) - duration)
        offset = np.random.randint(max_offset)
        data = np.pad(data, (offset, duration - len(data) - offset), "constant")

    elif len(data) > sample_rate:
        max_offset = np.abs(len(data) - duration)
        offset = np.random.randint(max_offset)
        data = data[offset:len(data) - max_offset + offset]

    else:
        offset = 0

    
    S_noisy = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))[:-1, :]
    S_noisy = np.expand_dims(S_noisy, -1)

    return S_noisy, data  


class ZeroOneNorm:
    
    def __init__(self):
        self.data_to_fit = []

   
    def fit(self, data_to_fit):
        self.fitting_constant = np.max(np.abs(data_to_fit))

    
    def normalize(self, data_to_transform):
        normalized_data = data_to_transform / self.fitting_constant
        return normalized_data

    #Denormalize:
    def denormalize(self, data_to_transform):
        denormalized_data = data_to_transform * self.fitting_constant
        return denormalized_data

def spec_plot(spectrogram, ax_index, title):
    spectrogram = spectrogram.reshape((1024, 44))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), ax=ax[ax_index])
    ax[ax_index].set_title(title)

#Funciones streamlit st para empezar el app
st.title("App Denoiser")
st.write("Suba su archivo ruidoso, para reconstruirlo utilizando nuestros modelos")


uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    #cargar archivo
    S_noisy, data_noisy = get_spectrogram_features_noisy(uploaded_file, sample_rate=22050)
    
    # Display
    st.audio(uploaded_file, format="audio/wav")
    
    #normalizar
    normalizer = ZeroOneNorm()
    normalizer.fit(S_noisy)
    S_noisy_n = normalizer.normalize(S_noisy)

    #col1 y col2 para separar app en dos columnas
    col1, col2 = st.columns(2)

    #columna para el modelo autoencoder
    with col1:
        st.subheader("Autoencoder Model")

        
        denoised_spectrogram_autoencoder = autoencoder_model.predict(np.expand_dims(S_noisy_n, 0))

        
        denoised_spectrogram_autoencoder = normalizer.denormalize(denoised_spectrogram_autoencoder)

        
        denoised_waveform_autoencoder = librosa.griffinlim(denoised_spectrogram_autoencoder.squeeze(), hop_length=hop_length)

        
        sf.write('denoised_output_autoencoder.wav', denoised_waveform_autoencoder, 22050)
        st.write("Audio reconstruido (Autoencoder)")
        st.audio('denoised_output_autoencoder.wav', format="audio/wav")

        
        show_spectrograms_auto = st.checkbox("Show Spectrograms (Autoencoder)")
        show_waveforms_auto = st.checkbox("Show Waveforms (Autoencoder)")

        
        if show_spectrograms_auto:
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            spec_plot(S_noisy, 0, 'Noisy Spectrogram')
            spec_plot(denoised_spectrogram_autoencoder, 1, 'Denoised Spectrogram (Autoencoder)')
            st.pyplot(fig)

        
        if show_waveforms_auto:
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            ax[0].plot(data_noisy)
            ax[0].set_title('Senal ruidosa original')
            ax[1].plot(denoised_waveform_autoencoder)
            ax[1].set_title('Senal limpia(Autoencoder)')
            st.pyplot(fig)

    #col2 para el modelo unet
    with col2:
        st.subheader("U-Net Model")

        
        denoised_spectrogram_unet = unet_model.predict(np.expand_dims(S_noisy_n, 0))

        
        denoised_spectrogram_unet = normalizer.denormalize(denoised_spectrogram_unet)

        
        denoised_waveform_unet = librosa.griffinlim(denoised_spectrogram_unet.squeeze(), hop_length=hop_length)

        
        sf.write('denoised_output_unet.wav', denoised_waveform_unet, 22050)
        st.write("Audio reconstruido (U-Net)")
        st.audio('denoised_output_unet.wav', format="audio/wav")

        #botones para mostra specs
        show_spectrograms_unet = st.checkbox("Mostrar espectogramas (U-Net)")
        show_waveforms_unet = st.checkbox("Mostrar Waveforms (U-Net)")

        
        if show_spectrograms_unet:
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            spec_plot(S_noisy, 0, 'Noisy Spectrogram')
            spec_plot(denoised_spectrogram_unet, 1, 'Denoised Spectrogram (U-Net)')
            st.pyplot(fig)

        
        if show_waveforms_unet:
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            ax[0].plot(data_noisy)
            ax[0].set_title('Senal de Audio original')
            ax[1].plot(denoised_waveform_unet)
            ax[1].set_title('Senal de audio reconstruida (U-Net)')
            st.pyplot(fig)


    


else:
    st.title("Modelo Denoiser de Facebook")
    st.write("Suba su archivo ruidoso para visualizar la reconstrucción con el modelo Denoiser de Facebook.")

    #cargar el modelo preentrenado de Facebook
    model = pretrained.dns64().eval()

    #subida del archivo de audio ruidoso
    uploaded_file = st.file_uploader("Seleccione un archivo de audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        #cargar el audio ruidoso
        noisy_waveform, sample_rate = torchaudio.load(uploaded_file)
        st.audio(uploaded_file, format="audio/wav")

        #formato correcto de waveform
        noisy_waveform = convert_audio(noisy_waveform, sample_rate, model.sample_rate, model.chin)

        # Aplicar el modelo Denoiser
        with torch.no_grad():
            denoised_waveform = model(noisy_waveform[None])[0]

        #guardar la señal reconstruida
        denoised_audio_path = "denoised_audio_facebook.wav"
        torchaudio.save(denoised_audio_path, denoised_waveform.cpu(), model.sample_rate)

        #mostrar el audio denoised
        st.write("Audio reconstruido con el modelo Denoiser de Facebook:")
        st.audio(denoised_audio_path, format="audio/wav")

        # Mostrar espectrogramas
        show_spectrograms = st.checkbox("Mostrar espectrogramas")
        if show_spectrograms:
            # Convertir a numpy para visualización
            noisy_waveform_np = noisy_waveform.squeeze().cpu().numpy()
            denoised_waveform_np = denoised_waveform.squeeze().cpu().numpy()

            # Calcular especs
            noisy_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_waveform_np)), ref=np.max)
            denoised_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_waveform_np)), ref=np.max)

            # Graficar espec
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            librosa.display.specshow(noisy_spectrogram, sr=model.sample_rate, x_axis="time", y_axis="log", cmap="viridis", ax=ax[0])
            ax[0].set_title("Espectrograma de audio ruidoso")
            librosa.display.specshow(denoised_spectrogram, sr=model.sample_rate, x_axis="time", y_axis="log", cmap="viridis", ax=ax[1])
            ax[1].set_title("Espectrograma de audio denoised")
            st.pyplot(fig)

        # Mostrar formas de onda
        show_waveforms = st.checkbox("Mostrar formas de onda")
        if show_waveforms:
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            ax[0].plot(noisy_waveform_np)
            ax[0].set_title("Forma de onda original (ruidosa)")
            ax[1].plot(denoised_waveform_np)
            ax[1].set_title("Forma de onda reconstruida (Denoiser)")
            st.pyplot(fig)
