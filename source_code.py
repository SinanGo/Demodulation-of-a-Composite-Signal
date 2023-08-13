import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift, ifft, ifftshift
import soundfile as sf

fs = 96000 # sample rate
seconds = 3 # recording duration
modulation_frequency = 5000

my_recording = np.loadtxt('81.txt')
time = np.linspace(0,seconds,my_recording.shape[0]) # start, stop, sample count

plt.figure(1) # composite signal in time domain
plt.plot(time, my_recording)

frequencies = np.linspace(-fs/2,fs/2,fs*seconds)
my_recording_in_frequency = fftshift(fft(my_recording))

plt.figure(2) # composite signal in frequency domain
plt.plot(frequencies, np.abs(my_recording_in_frequency))

bandpass_filter = np.where(((frequencies >= (-modulation_frequency-3000)) & (frequencies <= (-modulation_frequency+3000))) | 
                           ((frequencies >= (modulation_frequency-3000)) & (frequencies <= modulation_frequency+3000)),1,0)
bandpass_filtered_record = np.multiply(my_recording_in_frequency, bandpass_filter)

plt.figure(3) # bandpass filter
plt.plot(frequencies, bandpass_filter)

plt.figure(4) # bandpass filtered recording
plt.plot(frequencies, np.abs(bandpass_filtered_record))

#stage2 : multiply with cos(2*pi*f*t) in time domain
bandpass_filtered_record_in_time = ifft(ifftshift(bandpass_filtered_record))
bandpass_filtered_record_stage2 = np.multiply(bandpass_filtered_record_in_time,np.cos(2*np.pi*modulation_frequency*time))
bandpass_filtered_record_stage2_in_frequency = fftshift(fft(bandpass_filtered_record_stage2))

plt.figure(5)
plt.plot(frequencies, np.abs(bandpass_filtered_record_stage2_in_frequency))

lowpass_filter = np.where((frequencies >= -3000) & (frequencies <= 3000),1,0)
lowpass_filtered_record = np.multiply(lowpass_filter,bandpass_filtered_record_stage2_in_frequency)

plt.figure(6) # lowpass filter
plt.plot(frequencies, lowpass_filter)

plt.figure(7) # lowpass filtered recording
plt.plot(frequencies, np.abs(lowpass_filtered_record))

sf.write("5k.wav", np.abs(ifft(lowpass_filtered_record)), 96000, subtype= 'PCM_16')

plt.show()


