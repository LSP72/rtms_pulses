import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
# import mat73 # needed if .mat was not saved w/ -v7 option, 
             # if so, it is even best to reload in matlab,
             # and save it again with that -v7 option.

from scipy.signal import butter, sosfiltfilt



def data_dict(rec, num_imu=5):
    freq_list = [rec[v][freq] for v in rec.keys() for freq in ["FreqS"] if v.startswith("Acc_") or v.startswith("Gyro_")]
    if all(freq == freq_list[0] for freq in freq_list):
        return {
            'freq': freq_list[0],
            'acc_x': rec[f"Acc_{num_imu}"]["dat"],
            # 'acc_x_Fs': rec[f"Acc_{num_imu}"]["FreqS"],
            'acc_y': rec[f"Acc_{num_imu+1}"]["dat"],
            # 'acc_y_Fs': rec[f"Acc_{num_imu+1}"]["FreqS"],
            'acc_z': rec[f"Acc_{num_imu+2}"]["dat"],
            # 'acc_z_Fs': rec[f"Acc_{num_imu+2}"]["FreqS"],
            'gyro_x': rec[f"Gyro_{num_imu+3}"]["dat"],
            # 'gyro_x_Fs': rec[f"Gyro_{num_imu+3}"]["FreqS"],
            'gyro_y': rec[f"Gyro_{num_imu+4}"]["dat"],
            # 'gyro_y_Fs': rec[f"Gyro_{num_imu+4}"]["FreqS"],
            'gyro_z': rec[f"Gyro_{num_imu+5}"]["dat"],
            # 'gyro_z_Fs': rec[f"Gyro_{num_imu+5}"]["FreqS"],
        }
    else:
        print("Sampling frequencies are not the same for all IMUs.")
        return {
            'freq': freq_list[0],
            'acc_x': rec[f"Acc_{num_imu}"]["dat"],
            'acc_x_Fs': rec[f"Acc_{num_imu}"]["FreqS"],
            'acc_y': rec[f"Acc_{num_imu+1}"]["dat"],
            'acc_y_Fs': rec[f"Acc_{num_imu+1}"]["FreqS"],
            'acc_z': rec[f"Acc_{num_imu+2}"]["dat"],
            'acc_z_Fs': rec[f"Acc_{num_imu+2}"]["FreqS"],
            'gyro_x': rec[f"Gyro_{num_imu+3}"]["dat"],
            'gyro_x_Fs': rec[f"Gyro_{num_imu+3}"]["FreqS"],
            'gyro_y': rec[f"Gyro_{num_imu+4}"]["dat"],
            'gyro_y_Fs': rec[f"Gyro_{num_imu+4}"]["FreqS"],
            'gyro_z': rec[f"Gyro_{num_imu+5}"]["dat"],
            'gyro_z_Fs': rec[f"Gyro_{num_imu+5}"]["FreqS"],
        }


def _gyro_bias(gyro_data, Fs, static_sec=3):
    """
    Remove the bias from gyroscope data by subtracting the mean value.
    
    Parameters:
    gyro_data (numpy.ndarray): The gyroscope data array.
    Fs (float): The sampling frequency.
    static_sec (float): The duration of the static period in seconds.
    
    Returns:
    numpy.ndarray: The gyroscope data with bias removed.
    """
    n = int(static_sec * Fs)
    bias = np.mean(gyro_data[:n])
    return gyro_data - bias

def remove_gyro_bias(gyro, static_sec=3):
    """
    Remove the bias from gyroscope data by subtracting the mean value.
    
    Parameters:
    gyro_data (numpy.ndarray): The gyroscope data array.
    
    Returns:
    numpy.ndarray: The gyroscope data with bias removed.
    """
    var = [k for k in gyro.keys() if k.startswith("gyro")]
    if 'freq' in gyro.keys():
        freq = gyro['freq']
    else:
        freq = None  # Assuming all gyro data have the same sampling frequency
    for v in var:
        gyro_data = gyro[v]
        if freq is not None:
            gyro[f"{v}_bias_removed"] = _gyro_bias(gyro_data, freq, static_sec)
        else:
            gyro[f"{v}_bias_removed"] = _gyro_bias(gyro_data, gyro[f"{v}_Fs"], static_sec)

    return gyro

def _acc_tilt(acc_x, acc_y, Fs, static_sec=3):
    """
    Calculate the tilt angle from accelerometer data.
    
    Parameters:
    acc_x (numpy.ndarray): The x-component of the accelerometer data array.
    acc_y (numpy.ndarray): The y-component of the accelerometer data array.
    Fs (float): The sampling frequency.
    static_sec (float): The duration of the static period in seconds.
    
    Returns:
    numpy.ndarray: The tilt angle in degrees.
    """
    
    # Calculate tilt angle using arctangent
    
    return np.degrees(np.arctan2(acc_y, acc_x))

def tilt_from_acc(acc, static_sec=3):
    var = [k for k in acc.keys() if k.startswith("acc")]
    if 'freq' in acc.keys():
        freq = acc['freq']
        acc["acc_tilt"] = _acc_tilt(acc['acc_x'], acc['acc_y'], freq, static_sec)
    else:
        freq = None  # Assuming all gyro data have the same sampling frequency
        acc["acc_tilt"] = _acc_tilt(acc['acc_x'], acc['acc_y'], acc[f"{v}_Fs"], static_sec)

    return acc

def complementary_filter(acc, gyr_z, fs, gain=0.98, g=9.81, gate=0.15):
    """Estimate the sagittal tilt of one segment by fusing accelerometer and gyroscope.

    Assumed sensor axes: x along the segment, y forward-backward,
    z mediolateral (the axis the segment rotates about).

    Parameters
    ----------
    acc : np.ndarray, shape (N, 3)
        Accelerometer signal in m/s^2. Only the x and y columns are used for the tilt.
    gyr_z : np.ndarray, shape (N,)
        Gyroscope signal about the z axis, in deg/s (NOT rad/s), with the bias
        already removed (see remove_gyro_bias).
    fs : float
        Sampling rate in Hz.
    gain : float, optional
        Weight given to the gyroscope prediction (0 to 1), by default 0.98.
        Higher = smoother but more drift; lower = more accelerometer noise.
    g : float, optional
        Gravity in m/s^2, by default 9.81.
    gate : float, optional
        Maximum relative deviation of |acc| from g for the accelerometer to be
        used at a sample, by default 0.15 (= 15%).

    Returns
    -------
    np.ndarray, shape (N,)
        Segment tilt angle in degrees, one value per sample.
    """
    
    dt = 1 / fs
    acc_angles = np.degrees(np.arctan2(acc[:, 1], acc[:, 0]))  # array version, not the dict function
    acc_norm = np.linalg.norm(acc, axis=1)
    angles = np.empty(len(gyr_z)) # Initialize an empty array to store the angle values one by one
    angles[0] = acc_angles[0] # Initialize the first angle value with the accelerometer first angle value
    for k in range(1, len(gyr_z)): # loop starts at index 1 since the first angle is already initialized by the acc value
        predicted_angle = angles[k - 1] + gyr_z[k] * dt # prediction of angle from the previous angle and the gyroscope angular velocity
        if abs(acc_norm[k] - g) / g < gate: # if the accelerometer norm is within the gate threshold, we can trust the accelerometer angle
            angles[k] = gain * predicted_angle + (1 - gain) * acc_angles[k]
        else: # if not, we only use the predicted angle from the gyroscope
            angles[k] = predicted_angle 

    return angles



def main():
    data_path = "/Users/mathildetardif/Documents/Python/etude_3/data/2026.10.28_test_IMUs_v7.mat"
    raw = loadmat(data_path, simplify_cells=True)
    var = [k for k in raw.keys() if not k.startswith("__")]
    print(var)
    rec = raw[var[0]]

    electrode_1 = data_dict(rec, num_imu=5)
    fs = float(electrode_1['freq'])
    
    acc_1 = {'freq': fs, **{k: v for k, v in electrode_1.items() if k.startswith("acc")}}
    x = np.linspace(0, len(electrode_1['acc_x'])*(1/fs), len(electrode_1['acc_x']))
    gyro_1 = {'freq': fs, **{k: v for k, v in electrode_1.items() if k.startswith("gyro")}}
    gyro_1 = remove_gyro_bias(gyro_1, static_sec=3)

    fig2 = plt.figure(figsize=(10, 6))

    plt.plot(x[:len(x)//2], gyro_1['gyro_x'][:len(x)//2], label="Raw signal", alpha=0.7, color="blue")
    plt.plot(x[:len(x)//2], gyro_1['gyro_x_bias_removed'][:len(x)//2], label="Filtered signal", alpha=0.7, color="darkmagenta")
    plt.title("Gyro: Raw vs Filtered signal")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.show()

    acc_1_w_tilt = tilt_from_acc(acc_1, static_sec=3)

    fig3 = plt.figure(figsize=(10, 6))

    plt.plot(x[:len(x)//2], acc_1['acc_x'][:len(x)//2], label="Raw signal", alpha=0.7, color="blue")
    plt.plot(x[:len(x)//2], acc_1_w_tilt['acc_tilt'][:len(x)//2], label="Filtered signal", alpha=0.7, color="darkmagenta")
    plt.title("Accelerometer: Raw vs Filtered signal")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.show()


    # columns: along the shank (pointing up), forward-backward, remaining axis
    acc = np.column_stack([-acc_1['acc_y'], acc_1['acc_z'], acc_1['acc_x']])
    gyr = np.asarray(gyro_1['gyro_x_bias_removed'], dtype=float)  # rotation axis: check it is gyro_x

    n = int(3 * fs)
    print("Mean |acc| at rest:", np.linalg.norm(acc[:n], axis=1).mean())  # ~9.81 -> g=9.81 ; ~1 -> g=1.0

    angles = complementary_filter(acc, gyr, fs, gain=0.98, g=9.81, gate=0.15)

    t = np.arange(len(angles)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, angles)
    plt.xlabel("Time (s)")
    plt.ylabel("Shank tilt (deg)")
    plt.grid()
    plt.show()

    print("done")


if __name__ == "__main__":
    main()