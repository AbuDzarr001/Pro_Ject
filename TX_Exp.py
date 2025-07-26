import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd

def zadoff_chu_code(length=31, root=1):
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)

def generate_zc_wave(zc_seq, fc, fs, symbol_duration):
    t = np.arange(0, symbol_duration, 1/fs)
    wave = []
    for val in zc_seq:
        wave.extend(np.real(val) * np.cos(2*np.pi*fc*t) + np.imag(val) * np.sin(2*np.pi*fc*t))
    return np.array(wave)

def qpsk_modulate(bits):
    if len(bits) % 2 != 0:
        bits.append(0)
    constellation = {
        (0, 0): 1 + 1j,
        (0, 1): 1 - 1j,
        (1, 0): -1 + 1j,
        (1, 1): -1 - 1j
    }
    symbols = [constellation[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)]
    return np.array(symbols)

def bits_to_wave(bits, fc, fs, symbol_duration):
    symbols = qpsk_modulate(bits)
    t = np.arange(0, symbol_duration, 1/fs)
    wave = []
    window = np.hamming(len(t))
    for sym in symbols:
        i_component = np.real(sym) * np.cos(2 * np.pi * fc * t)
        q_component = np.imag(sym) * np.sin(2 * np.pi * fc * t)
        symbol_wave = (i_component + q_component) * window
        wave.extend(symbol_wave)
    return np.array(wave)

def hamming_encode(bits):
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    if len(bits) % 4 != 0:
        bits += [0] * (4 - len(bits) % 4)
    groups = np.array(bits).reshape(-1, 4)
    encoded = np.dot(groups, G) % 2
    return encoded.flatten()

# Compute signal power
def compute_power(signal):
    return np.mean(np.abs(signal)**2)

def transmitter(message, fc=3000, fs=44100, symbol_duration=0.02):
    print("[Transmitter] Encoding message...")
    bitstream = []
    for char in message:
        bits = format(ord(char), '08b')
        bitstream.extend([int(b) for b in bits])

    bitstream = hamming_encode(bitstream)

    if len(bitstream) % 2 != 0:
        bitstream.append(0)

    print(f"[Debug] Total bits (Hamming encoded): {len(bitstream)}")
    print(f"[Debug] Total QPSK symbols: {len(bitstream)//2}")

    zc_len = 128
    zc_seq_pre = zadoff_chu_code(length=zc_len, root=1)
    np.save("zc_symbols__towingtank.npy", zc_seq_pre)
    zc_seq_post = zadoff_chu_code(length=zc_len, root=2)

    zc_wave_pre = generate_zc_wave(zc_seq_pre, fc, fs, symbol_duration)
    zc_wave_post = generate_zc_wave(zc_seq_post, fc, fs, symbol_duration)

    waveform = bits_to_wave(bitstream, fc, fs, symbol_duration)

    tx_wave = np.concatenate([zc_wave_pre, waveform, zc_wave_post])

    # Daya sebelum normalisasi
    power_before_norm = compute_power(tx_wave)
    print(f"[Transmitter] Power before normalization: {power_before_norm:.6f}")
    tx_wave = tx_wave / np.max(np.abs(tx_wave))
    # Daya setelah normalisasi
    power_after_norm = compute_power(tx_wave)
    print(f"[Transmitter] Power after normalization:  {power_after_norm:.6f}")

    filename = "transmitted_signal_kabel_5000_6000fs.wav"
    sf.write(filename, tx_wave, fs)
    print(f"[Transmitter] Signal saved to '{filename}'")

    #sd.play(tx_wave, fs)
    #sd.wait()

    plt.figure(figsize=(12, 4))
    plt.plot(tx_wave)
    plt.title("Transmitter Signal (Time Domain)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('transmitter_no_header_signal.png')
    plt.show()
    
    plt.figure()
    plt.magnitude_spectrum(tx_wave, Fs=fs, scale='dB')
    plt.title("Spektrum Sinyal Masuk (Receiver)")
    plt.grid(True)
    plt.show()

    #np.save("original_bits_Exp_48k.npy", bitstream)  # bit_stream sebelum mapping QPSK
    
    symbols = qpsk_modulate(bitstream)
    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(symbols), np.imag(symbols), color='blue', s=10)
    plt.title("QPSK Constellation Diagram")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('transmitter_no_header_constellation.png')
    plt.show()
    
    

    return filename

# --- MAIN ---
if __name__ == "__main__":
    message = "Alpha123 Bravo456 Charlie789 Test01!"
    print("[Transmitter] Message to be sent:", message)
    transmitter(message)
