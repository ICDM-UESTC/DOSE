from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa

from pystoi import stoi

from dose_plus.util.other import energy_ratios, mean_std
from dose_plus.util.semp import composite


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "csig": [], "cbak": [],  "covl": []}

    # Evaluate standard metrics
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        x, sr_x = read(join(args.clean_dir, clean_filename))
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))
        assert sr_x == sr_y == sr_x_hat
        n = y - x 
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        semp = composite(x,x_hat, sr_x)
        _, _, _csig, _cbak, _covl, _ = semp(x, x_hat, sr_x)
        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        data["csig"].append(_csig)
        data["cbak"].append(_cbak)
        data["covl"].append(_covl)

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("CSIG: {:.2f} ± {:.2f}".format(*mean_std(df["csig"].to_numpy())))
    print("CBAK: {:.2f} ± {:.2f}".format(*mean_std(df["cbak"].to_numpy())))
    print("COVL: {:.2f} ± {:.2f}".format(*mean_std(df["covl"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("CSIG: {:.2f} ± {:.2f}".format(*mean_std(df["csig"].to_numpy())) + "\n")
    log.write("CBAK: {:.2f} ± {:.2f}".format(*mean_std(df["cbak"].to_numpy())) + "\n")
    log.write("COVL: {:.2f} ± {:.2f}".format(*mean_std(df["covl"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)
