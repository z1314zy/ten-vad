import sys
import os
from pathlib import Path
from math import gcd

import numpy as np
import scipy.io.wavfile as Wavfile
from scipy.signal import resample_poly

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad

INPUT_DIR = Path("your_path")

HOP_SIZE = 256          # 16kHz 下 256 samples = 16 ms
THRESHOLD = 0.5

TARGET_SR = 16000

AUDIO_SUFFIXES = {".wav", ".flac"}

# 是否允许自动重采样到 16k
ALLOW_RESAMPLE = True


def prob_to_segments(prob_list, threshold, hop_size, sample_rate):

    frame_shift_s = hop_size / float(sample_rate)

    binary = []
    for prob in prob_list:
        if prob >= threshold:
            binary.append(1)
        else:
            binary.append(0)

    segments = []
    in_speech = False
    start_frame = 0

    for i, flag in enumerate(binary):
        if flag == 1 and not in_speech:
            in_speech = True
            start_frame = i

        elif flag == 0 and in_speech:
            end_frame = i

            start_s = start_frame * frame_shift_s
            end_s = end_frame * frame_shift_s

            segments.append([start_s, end_s])
            in_speech = False

    if in_speech:
        end_frame = len(binary)

        start_s = start_frame * frame_shift_s
        end_s = end_frame * frame_shift_s

        segments.append([start_s, end_s])

    return segments, binary



def get_dir_txt_prefix(root: Path):

    txt_files = []

    for p in root.iterdir():
        if not p.is_file():
            continue

        if p.suffix.lower() != ".txt":
            continue

        name_lower = p.name.lower()

        if name_lower.endswith("_vad.txt"):
            continue

        if name_lower.endswith("_fail.txt"):
            continue

        if name_lower == "fail.txt":
            continue

        txt_files.append(p)

    if len(txt_files) > 0:
        txt_files = sorted(txt_files, key=lambda x: x.name)
        return txt_files[0].stem

    return root.name

def read_by_soundfile(audio_path: Path):

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "当前环境没有安装 soundfile，无法稳定读取 flac。\n"
            "请先安装：pip install soundfile"
        )

    data, sr = sf.read(str(audio_path), always_2d=False)
    return sr, data


def convert_audio_to_float32(data):

    data = np.asarray(data)

    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0

    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    else:
        data = data.astype(np.float32)

    data = np.nan_to_num(data)
    data = np.clip(data, -1.0, 1.0)

    return data


def float32_to_int16(data_float):
    """
    float32 [-1, 1] -> int16
    """

    data_float = np.asarray(data_float, dtype=np.float32)
    data_float = np.nan_to_num(data_float)
    data_float = np.clip(data_float, -1.0, 1.0)

    data_int16 = (data_float * 32767.0).astype(np.int16)

    return data_int16


def resample_audio_if_needed(data_float, sr, target_sr=16000):

    if sr == target_sr:
        return data_float, sr

    if not ALLOW_RESAMPLE:
        raise ValueError(f"TEN-VAD 只支持 16kHz，当前采样率为 {sr}")

    g = gcd(int(sr), int(target_sr))
    up = target_sr // g
    down = sr // g

    data_float = resample_poly(data_float, up, down).astype(np.float32)

    return data_float, target_sr


def read_audio_as_16k_mono_int16(audio_path: Path, target_sr=16000):

    audio_path = Path(audio_path)
    suffix = audio_path.suffix.lower()

    if suffix not in AUDIO_SUFFIXES:
        raise ValueError(f"不支持的音频格式: {suffix}")

    sr = None
    data = None

    if suffix == ".wav":
        try:
            sr, data = Wavfile.read(str(audio_path))
        except Exception:
            sr, data = read_by_soundfile(audio_path)

    elif suffix == ".flac":
        sr, data = read_by_soundfile(audio_path)

    data_float = convert_audio_to_float32(data)
    data_float, sr = resample_audio_if_needed(
        data_float=data_float,
        sr=sr,
        target_sr=target_sr
    )

    data_int16 = float32_to_int16(data_float)

    return sr, data_int16

def run_ten_vad_on_audio(audio_path: Path, hop_size=256, threshold=0.5):

    sr, data = read_audio_as_16k_mono_int16(
        audio_path=audio_path,
        target_sr=TARGET_SR
    )

    if sr != TARGET_SR:
        raise ValueError(f"TEN-VAD 只支持 16kHz，当前采样率为 {sr}")

    if data.ndim != 1:
        raise ValueError(f"音频不是单通道，当前 shape={data.shape}")

    if data.dtype != np.int16:
        raise ValueError(f"音频不是 int16，当前 dtype={data.dtype}")

    ten_vad_instance = TenVad(hop_size, threshold)

    num_frames = data.shape[0] // hop_size
    prob_list = []

    for i in range(num_frames):
        audio_frame = data[i * hop_size: (i + 1) * hop_size]
        out_probability, _ = ten_vad_instance.process(audio_frame)
        prob_list.append(float(out_probability))

    del ten_vad_instance

    segments, binary = prob_to_segments(
        prob_list=prob_list,
        threshold=threshold,
        hop_size=hop_size,
        sample_rate=sr
    )

    return segments, prob_list, binary

def process_one_audio_dir(root: Path, audio_files, hop_size=256, threshold=0.5):

    prefix = get_dir_txt_prefix(root)

    output_txt = root / f"{prefix}_VAD.txt"
    fail_txt = root / f"{prefix}_fail.txt"

    success_count = 0
    failed_files = []

    with open(output_txt, "w", encoding="utf-8") as out_f:
        for audio_path in sorted(audio_files, key=lambda x: x.name):
            abs_audio_path = audio_path.resolve()

            try:
                segments, prob_list, binary = run_ten_vad_on_audio(
                    audio_path=abs_audio_path,
                    hop_size=hop_size,
                    threshold=threshold
                )

                file_no_ext = audio_path.stem

                if len(segments) > 0:
                    seg_strs = []
                    for start_s, end_s in segments:
                        seg_strs.append(f"({start_s:.3f}, {end_s:.3f})")

                    line = f"{file_no_ext} {', '.join(seg_strs)}\n"
                    out_f.write(line)
                else:
                    out_f.write(f"{file_no_ext}\n")

                success_count += 1

            except Exception as e:
                err_msg = str(e)
                print(f"[失败] {abs_audio_path}: {err_msg}")
                failed_files.append((str(abs_audio_path), err_msg))
                continue

    if len(failed_files) > 0:
        with open(fail_txt, "w", encoding="utf-8") as f_fail:
            for fail_file, err_msg in failed_files:
                f_fail.write(f"{fail_file}\t{err_msg}\n")
        fail_txt_return = str(fail_txt.resolve())
    else:
        if fail_txt.exists():
            fail_txt.unlink()
        fail_txt_return = None

    return {
        "root": str(root.resolve()),
        "output_txt": str(output_txt.resolve()),
        "fail_txt": fail_txt_return,
        "total": len(audio_files),
        "success": success_count,
        "failed": len(failed_files),
    }

def main():
    input_dir = INPUT_DIR

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    all_dir_results = []

    total_audio_dirs = 0
    total_audio_count = 0
    total_success_count = 0
    total_failed_count = 0

    for root, dirs, files in os.walk(input_dir):
        root = Path(root)

        audio_files = []

        for file in files:
            p = root / file

            if p.suffix.lower() in AUDIO_SUFFIXES:
                audio_files.append(p)

        if len(audio_files) == 0:
            continue

        total_audio_dirs += 1

        print("\n" + "-" * 80)
        print(f"正在处理目录: {root}")
        print(f"当前目录音频数量: {len(audio_files)}")

        result = process_one_audio_dir(
            root=root,
            audio_files=audio_files,
            hop_size=HOP_SIZE,
            threshold=THRESHOLD
        )

        all_dir_results.append(result)

        total_audio_count += result["total"]
        total_success_count += result["success"]
        total_failed_count += result["failed"]

        print(f"结果文件: {result['output_txt']}")

        if result["fail_txt"] is not None:
            print(f"失败文件: {result['fail_txt']}")
        else:
            print("失败文件: 无失败音频，未生成")

        print(f"成功: {result['success']} / {result['total']}")
        print(f"失败: {result['failed']} / {result['total']}")

    print("\n" + "=" * 80)
    print("TEN-VAD 批量处理完成")
    print("=" * 80)
    print(f"输入根目录: {input_dir.resolve()}")
    print(f"包含音频的目录数量: {total_audio_dirs}")
    print(f"总音频数: {total_audio_count}")
    print(f"成功处理音频数: {total_success_count}")
    print(f"处理失败音频数: {total_failed_count}")
    print("=" * 80)

    if len(all_dir_results) == 0:
        print("没有找到 wav/flac 音频文件。")
        return

    print("\n各目录输出汇总:")

    for r in all_dir_results:
        print("-" * 80)
        print(f"目录: {r['root']}")
        print(f"VAD结果: {r['output_txt']}")

        if r["fail_txt"] is not None:
            print(f"失败文件: {r['fail_txt']}")
        else:
            print("失败文件: 无失败音频，未生成")

        print(f"成功/总数: {r['success']} / {r['total']}")
        print(f"失败: {r['failed']}")


if __name__ == "__main__":
    main()