import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt

def zadoff_chu_code(length=128, root=1):
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)

def qpsk_demodulate(symbols):
    bits = []
    for sym in symbols:
        i = np.real(sym)
        q = np.imag(sym)
        bits.extend([int(i < 0), int(q < 0)])
    return bits

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(chars)

def lowpass_filter(signal, fs, cutoff, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def hamming_decode(bits):
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ])
    corrected = []
    bits = np.array(bits)
    if len(bits) % 7 != 0:
        return []
    groups = bits.reshape(-1, 7)
    for word in groups:
        syndrome = np.dot(H, word) % 2
        s_val = int("".join(str(x) for x in syndrome), 2)
        if 0 < s_val <= 7:
            word[s_val - 1] ^= 1
        corrected.extend(word[:4])
    return corrected

def estimate_channel(rx_wave, zc_seq, fc, fs, symbol_duration):
    t = np.arange(0, symbol_duration, 1/fs)
    est = []
    for i, val in enumerate(zc_seq):
        seg = rx_wave[i*len(t):(i+1)*len(t)]
        if len(seg) < len(t):
            break
        i_rx = 2 * np.sum(seg * np.cos(2 * np.pi * fc * t)) / len(t)
        q_rx = 2 * np.sum(seg * np.sin(2 * np.pi * fc * t)) / len(t)
        rx_sym = i_rx + 1j * q_rx
        est.append(rx_sym / val)
    return np.mean(est)

def extract_symbols(signal, fc, fs, symbol_duration):
    t = np.arange(0, symbol_duration, 1/fs)
    sym_len = len(t)
    symbols = []
    for i in range(0, len(signal) - sym_len + 1, sym_len):
        seg = signal[i:i+sym_len]
        i_comp = 2 * np.sum(seg * np.cos(2*np.pi*fc*t)) / sym_len
        q_comp = 2 * np.sum(seg * np.sin(2*np.pi*fc*t)) / sym_len
        symbols.append(i_comp + 1j * q_comp)
    return np.array(symbols)

def compute_power(signal):
    return np.mean(np.abs(signal)**2)

def analyze_sync_metrics(corr, peak_index, label="Preamble", fs=44100):
    import matplotlib.pyplot as plt
    import numpy as np

    # Hitung sidelobe dengan mengabaikan ±len//2 dari puncak
    window = len(corr) // 10  # daerah di sekitar puncak untuk diabaikan
    sidelobe = np.concatenate([
        np.abs(corr[:max(0, peak_index - window)]),
        np.abs(corr[min(len(corr), peak_index + window):])
    ])
    mean_sidelobe = np.mean(sidelobe)
    peak_value = np.abs(corr[peak_index])
    psr = peak_value / (mean_sidelobe + 1e-12)

    print(f"[Sync-{label}] Peak index     : {peak_index}")
    print(f"[Sync-{label}] Peak amplitude : {peak_value:.2f}")
    print(f"[Sync-{label}] Mean sidelobe  : {mean_sidelobe:.2f}")
    print(f"[Sync-{label}] Peak-to-sidelobe ratio (PSR): {psr:.2f}")

    # Plot korelasi
    t = np.arange(len(corr)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, np.abs(corr))
    plt.axvline(t[peak_index], color='r', linestyle='--', label='Peak')
    plt.title(f"Korelasi dengan {label} (PSR = {psr:.2f})")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Magnitudo Korelasi")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def receiver(filename, fc=3000, fs=44100, symbol_duration=0.02, cutoff_freq=3000):
    rx_wave, _ = sf.read(filename)
    rx_wave, _ = sf.read(filename)
    if rx_wave.ndim == 2:
        print(f"[Receiver] Input stereo dengan shape: {rx_wave.shape}")
        ch_L = compute_power(rx_wave[:, 0])
        ch_R = compute_power(rx_wave[:, 1])
        print(f"[Receiver] Power L: {ch_L:.4f}, Power R: {ch_R:.4f}")
        rx_wave = rx_wave.mean(axis=1)  # atau pilih yang paling besar

    power = compute_power(rx_wave)
    print(f"[Transmitter] Power: {power:.6f}")
    
    # Plot sinyal utuh (sebelum filtering)
    plt.figure(figsize=(12, 4))
    plt.plot(rx_wave)
    plt.title("Sinyal QPSK")
    plt.xlabel("Sampel")
    plt.ylabel("Amplitudo")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('D:/S2/Data Modem/Plot Gambar/Pengujian 3/50/sinyal_asli_3000_0,02.png')
    plt.show()
    
    plt.figure()
    plt.magnitude_spectrum(rx_wave, Fs=fs, scale='dB')
    plt.title("Spectrum Signal")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('D:/S2/Data Modem/Plot Gambar/Pengujian 3/50/receiver_asli_3000_0,02.png')
    plt.show()

    filtered_wave = rx_wave
    
    # Plot sinyal utuh (setelah filtering)
    plt.figure(figsize=(12, 4))
    plt.plot(filtered_wave)
    plt.title("Sinyal QPSK Setelah Filter")
    plt.xlabel("Sampel")
    plt.ylabel("Amplitudo")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('D:/S2/Data Modem/Plot Gambar/Pengujian 3/50/sinyal_asli_3000_0,02.png')
    plt.show()
    
    plt.figure()
    plt.magnitude_spectrum(filtered_wave, Fs=fs, scale='dB')
    plt.title("Spectrum Signal")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('D:/S2/Data Modem/Plot Gambar/Pengujian 3/50/receiver_asli_3000_0,02.png')
    plt.show()

    # Generate Zadoff-Chu preamble & postamble
    zc_len = 128
    zc_pre = zadoff_chu_code(zc_len, root=1)
    zc_post = zadoff_chu_code(zc_len, root=2)
    t = np.arange(0, symbol_duration, 1/fs)
    zc_wave_pre = np.concatenate([
        np.real(val)*np.cos(2*np.pi*fc*t) + np.imag(val)*np.sin(2*np.pi*fc*t)
        for val in zc_pre
    ])
    zc_wave_post = np.concatenate([
        np.real(val)*np.cos(2*np.pi*fc*t) + np.imag(val)*np.sin(2*np.pi*fc*t)
        for val in zc_post
    ])

    # Korelasi preamble
    corr_pre = correlate(filtered_wave, zc_wave_pre, mode='valid')
    peak_pre = np.argmax(np.abs(corr_pre))
    start = peak_pre + len(zc_wave_pre)

    # Analisis sinkronisasi preamble
    analyze_sync_metrics(corr_pre, peak_pre, label="Preamble", fs=fs)

    # Korelasi postamble setelah start
    search_start = start + int(0.1 * fs)
    corr_post = correlate(filtered_wave[search_start:], zc_wave_post, mode='valid')
    peak_post = np.argmax(np.abs(corr_post))
    end = search_start + peak_post

    # Analisis sinkronisasi postamble
    analyze_sync_metrics(corr_post, peak_post, label="Postamble", fs=fs)


    # Potong sinyal data
    data_wave = filtered_wave[start:end]
    symbol_len = int(symbol_duration * fs)
    valid_len = len(data_wave) - (len(data_wave) % symbol_len)
    data_wave = data_wave[:valid_len]
    
    # --- Zoom pada sebagian sinyal ---
    # Misal kita ingin melihat bagian tengah sinyal lebih dekat
    start_sample = int(len(data_wave) * 0.45)
    end_sample = int(len(data_wave) * 0.55)

    plt.figure(figsize=(12, 4))
    plt.plot(data_wave[start_sample:end_sample])
    plt.title('Zoomed Receiver Signal (Time Domain)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
        # --- Plot seluruh sinyal data dalam domain waktu ---
    t_full = np.arange(len(data_wave)) / fs  # waktu dalam detik
    plt.figure(figsize=(12, 4))
    plt.plot(t_full, np.real(data_wave), label='Real Part', alpha=0.7)
    plt.plot(t_full, np.imag(data_wave), label='Imag Part', alpha=0.7)
    plt.title("Sinyal Data dalam Domain Waktu (Keseluruhan)")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
        # --- Plot zoom sinyal domain waktu untuk beberapa simbol ---
    num_symbols_to_plot = 5
    samples_to_plot = num_symbols_to_plot * symbol_len

    if len(data_wave) >= samples_to_plot:
        zoom_signal = data_wave[:samples_to_plot]
        t = np.arange(samples_to_plot) / fs * 1000  # waktu dalam ms

        plt.figure(figsize=(10, 4))
        plt.plot(t, zoom_signal)
        plt.title(f"Sinyal Domain Waktu (Zoom {num_symbols_to_plot} Simbol)")
        plt.xlabel("Waktu (ms)")
        plt.ylabel("Amplitudo")
        for i in range(num_symbols_to_plot + 1):
            plt.axvline(x=i * symbol_duration * 1000, color='gray', linestyle=':', linewidth=1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("[Receiver] Sinyal terlalu pendek untuk zoom simbol.")

    
    # --- FFT Analysis for Frequency Offset Detection ---
    N = 8192  # Number of FFT points
    fft_data = data_wave[:N]
    fft_result = np.fft.fft(fft_data * np.hanning(len(fft_data)))
    freqs = np.fft.fftfreq(len(fft_result), 1/fs)
    fft_magnitude = np.abs(fft_result)

    # Hanya tampilkan frekuensi positif
    pos_freqs = freqs[:len(freqs)//2]
    pos_magnitude = fft_magnitude[:len(freqs)//2]

    peak_index = np.argmax(pos_magnitude)
    peak_freq = pos_freqs[peak_index]
    freq_offset = peak_freq - fc

    print(f"[Receiver] FFT peak frequency: {peak_freq:.2f} Hz")
    print(f"[Receiver] Detected frequency offset: {freq_offset:.2f} Hz")

    plt.figure(figsize=(10, 4))
    plt.plot(pos_freqs, pos_magnitude)
    plt.title("Spektrum Frekuensi Sinyal Data")
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Magnitudo")
    plt.grid(True)
    plt.axvline(fc, color='r', linestyle='--', label='fc')
    plt.axvline(peak_freq, color='g', linestyle='--', label='Peak')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Plot sinyal yang didekode (data_wave)
    plt.figure(figsize=(10, 4))
    plt.plot(data_wave)
    plt.title("Sinyal Setelah Preamble dan Sebelum Ekstraksi Simbol")
    plt.xlabel("Sampel")
    plt.ylabel("Amplitudo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Estimasi saluran
    preamble_segment = filtered_wave[start - len(zc_wave_pre):start]
    h_est = estimate_channel(preamble_segment, zc_pre, fc, fs, symbol_duration)
    print(f"[Receiver] Channel estimate: {h_est:.4f}")

    # Ekstrak simbol, equalize, dan demodulasi
    symbols = extract_symbols(data_wave, fc, fs, symbol_duration)
    symbols_eq = symbols / h_est
    bits = qpsk_demodulate(symbols_eq)

    print(f"[Receiver] Demodulated bits (first 40): {bits[:40]}")

   # Buat panjang bitstream kelipatan 7
    bits = bits[:len(bits) - (len(bits) % 7)]
    print(f"[Receiver] Total bits after trimming to multiple of 7: {len(bits)}")

    decoded_bits = hamming_decode(bits)
    #np.save("decoded_bits_asli.npy", bits)
    print(f"[Receiver] Decoded bits (first 40): {decoded_bits[:40]}")
    print("[Receiver] Decoded bits saved")
    
    message = bits_to_string(decoded_bits)

    # Plot konstelasi
    plt.figure(figsize=(6, 6))
    plt.plot(np.real(symbols_eq), np.imag(symbols_eq), 'o')
    plt.title("Konstelasi QPSK")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis('equal')
    #plt.savefig('D:/S2/Data Modem/Plot Gambar/Pengujian 3/50/Konstelasi_asli_3000_0,02.png')
    plt.show()

    print("[Receiver] Decoded message:", message)

    return message

# --- MAIN ---
if __name__ == "__main__":
    receiver("D:/S2/Data Modem/Kalibrasi/Hasil/Variasi/3000_44100.wav")
