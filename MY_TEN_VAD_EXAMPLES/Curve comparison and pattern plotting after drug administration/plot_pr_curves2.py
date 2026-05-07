#
#  Plot waveform and VAD results of Silero VAD, TEN VAD and FSMN VAD
#

import os
import sys
import glob
from pathlib import Path

import numpy as np
import torchaudio
import scipy.io.wavfile as Wavfile
import matplotlib.pyplot as plt


# =========================
# 1. 路径配置
# =========================

# Silero VAD 路径
sys.path.append(r"D:\VAD\1.vad_learn\silero-vad\src")
from silero_vad.utils_vad import VADIterator, init_jit_model

# FSMN VAD 路径
sys.path.append(r"D:\VAD\1.vad_learn\fsmn-vad\fsmnvad")
import fsmnvad

# TEN VAD 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad


# =========================
# 2. 音频读取与归一化
# =========================

def read_wav_for_plot(wav_path):
    """
    读取 wav，用于画输入波形。
    返回：
        sample_rate
        waveform_float: [-1, 1] 左右的单通道波形
    """
    sample_rate, data = Wavfile.read(wav_path)

    if len(data.shape) > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.integer):
        max_value = np.iinfo(data.dtype).max
        waveform = data.astype(np.float32) / max_value
    else:
        waveform = data.astype(np.float32)

    return sample_rate, waveform


# =========================
# 3. TEN VAD 推理
# =========================

def ten_vad_process_wav(wav_path, hop_size=256, threshold=0.5):
    """
    TEN VAD 输出逐帧 voice probability。
    """
    sample_rate, data = Wavfile.read(wav_path)

    assert sample_rate == 16000, f"TEN VAD only supports 16kHz, but got {sample_rate}: {wav_path}"

    if len(data.shape) > 1:
        data = data[:, 0]

    ten_vad_instance = TenVad(hop_size, threshold)

    num_frames = data.shape[0] // hop_size
    voice_prob_arr = np.array([], dtype=np.float32)

    for i in range(num_frames):
        input_data = data[i * hop_size: (i + 1) * hop_size]
        voice_prob, _ = ten_vad_instance.process(input_data)
        voice_prob_arr = np.append(voice_prob_arr, voice_prob)

    del ten_vad_instance

    return voice_prob_arr, hop_size


# =========================
# 4. Silero VAD 推理
# =========================

def silero_vad_process_wav(model, wav_path):
    """
    Silero VAD 输出逐帧 speech probability。
    """
    window_size_samples = 512
    speech_probs = np.array([], dtype=np.float32)

    wav, sr = torchaudio.load(wav_path)

    assert sr == 16000, f"Silero VAD only supports 16kHz, but got {sr}: {wav_path}"

    if wav.shape[0] > 1:
        wav = wav[0, :]
    else:
        wav = wav.squeeze(0)

    vad_iterator = VADIterator(model)

    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i + window_size_samples]

        if len(chunk) < window_size_samples:
            break

        speech_prob = model(chunk, sr).item()
        speech_probs = np.append(speech_probs, speech_prob)

    vad_iterator.reset_states()

    return speech_probs, window_size_samples


# =========================
# 5. FSMN VAD 推理
# =========================

def fsmn_vad_process_wav(fsmn_vad_instance, wav_path):
    """
    FSMN VAD 无后处理版本：
    直接取模型 raw scores，不经过 infer_offline / VAD 状态机 / 端点检测。

    要求：
        fsmnvad.py 中已经新增 raw_scores_offline()
    """
    if not hasattr(fsmn_vad_instance, "raw_scores_offline"):
        raise AttributeError(
            "当前 FSMNVad 没有 raw_scores_offline() 接口。\n"
            "请先在 fsmnvad.py 中新增 raw_scores_offline()。"
        )

    speech_scores = fsmn_vad_instance.raw_scores_offline(Path(wav_path))
    speech_scores = np.asarray(speech_scores, dtype=np.float32)

    sample_rate, wav_data = Wavfile.read(wav_path)

    if len(wav_data.shape) > 1:
        wav_len = wav_data.shape[0]
    else:
        wav_len = len(wav_data)

    if len(speech_scores) > 0:
        hop_size = int(round(wav_len / len(speech_scores)))
    else:
        hop_size = int(sample_rate * 0.01)

    return speech_scores, hop_size


# =========================
# 6. 将概率转换为 0/1 VAD 结果
# =========================

def prob_to_binary(prob_arr, threshold=0.5):
    """
    概率/分数转成二值 VAD 结果。
    """
    prob_arr = np.asarray(prob_arr, dtype=np.float32)
    return np.where(prob_arr >= threshold, 1.0, 0.0)


def make_step_time_and_value(binary_arr, hop_size, sample_rate, duration):
    """
    生成阶梯图时间轴。
    """
    binary_arr = np.asarray(binary_arr, dtype=np.float32)

    if len(binary_arr) == 0:
        return np.array([0.0, duration], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

    t = np.arange(len(binary_arr), dtype=np.float32) * hop_size / sample_rate

    # 追加结尾，保证阶梯图延伸到音频结束
    t = np.append(t, duration)
    y = np.append(binary_arr, binary_arr[-1])

    return t, y


# =========================
# 7. 绘制单条音频的四联图
# =========================

def plot_vad_comparison_for_one_file(
        wav_path,
        save_path,
        silero_probs,
        silero_hop,
        ten_probs,
        ten_hop,
        fsmn_probs,
        fsmn_hop,
        silero_threshold=0.5,
        ten_threshold=0.5,
        fsmn_threshold=0.5,
):
    """
    绘制：
        1. Input audio
        2. Silero VAD
        3. TEN VAD
        4. FSMN VAD
    """
    sample_rate, waveform = read_wav_for_plot(wav_path)
    duration = len(waveform) / sample_rate

    time_audio = np.arange(len(waveform), dtype=np.float32) / sample_rate

    silero_binary = prob_to_binary(silero_probs, silero_threshold)
    ten_binary = prob_to_binary(ten_probs, ten_threshold)
    fsmn_binary = prob_to_binary(fsmn_probs, fsmn_threshold)

    t_silero, y_silero = make_step_time_and_value(
        silero_binary,
        silero_hop,
        sample_rate,
        duration
    )

    t_ten, y_ten = make_step_time_and_value(
        ten_binary,
        ten_hop,
        sample_rate,
        duration
    )

    t_fsmn, y_fsmn = make_step_time_and_value(
        fsmn_binary,
        fsmn_hop,
        sample_rate,
        duration
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    # -------------------------
    # 1. 输入波形
    # -------------------------
    axes[0].plot(time_audio, waveform, linewidth=0.8)
    axes[0].set_title("Input audio", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Normalization Value", fontsize=12)
    axes[0].grid(False)

    max_abs = np.max(np.abs(waveform)) if len(waveform) > 0 else 1.0
    y_lim = max(0.1, max_abs * 1.2)
    axes[0].set_ylim(-y_lim, y_lim)

    # -------------------------
    # 2. Silero VAD
    # -------------------------
    axes[1].step(t_silero, y_silero, where="post", linewidth=1.2)
    axes[1].set_title("Silero VAD", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("VAD Result", fontsize=12)
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].grid(False)

    # -------------------------
    # 3. TEN VAD
    # -------------------------
    axes[2].step(t_ten, y_ten, where="post", linewidth=1.2)
    axes[2].set_title("TEN VAD", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("VAD Result", fontsize=12)
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].grid(False)

    # -------------------------
    # 4. FSMN VAD
    # -------------------------
    axes[3].step(t_fsmn, y_fsmn, where="post", linewidth=1.2)
    axes[3].set_title("FSMN VAD", fontsize=14, fontweight="bold")
    axes[3].set_ylabel("VAD Result", fontsize=12)
    axes[3].set_xlabel("Time in sec", fontsize=12)
    axes[3].set_ylim(-0.05, 1.1)
    axes[3].grid(False)

    for ax in axes:
        ax.set_xlim(0, duration)
        ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 8. 主程序
# =========================

if __name__ == "__main__":

    # 当前脚本目录，例如：
    # D:\VAD\1.vad_learn\ten-vad\examples
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 测试集目录
    test_dir = os.path.abspath(os.path.join(script_dir, "../testset"))

    # 保存目录
    save_dir = r"D:\VAD\1.vad_learn\ten-vad\save"
    os.makedirs(save_dir, exist_ok=True)

    # Silero 模型路径
    silero_model_path = r"D:\VAD\1.vad_learn\silero-vad\src\silero_vad\data\silero_vad.jit"

    # 参数
    sample_rate = 16000

    ten_hop_size = 256
    ten_threshold = 0.5

    silero_threshold = 0.5
    fsmn_threshold = 0.5

    # 获取 wav 文件
    wav_list = glob.glob(os.path.join(test_dir, "*.wav"))
    wav_list = sorted(wav_list)

    if len(wav_list) == 0:
        raise FileNotFoundError(f"No wav files found in: {test_dir}")

    print(f"Found {len(wav_list)} wav files.")
    print("Loading models...")

    # Silero 模型只加载一次
    silero_model = init_jit_model(silero_model_path)

    # FSMN 模型只加载一次
    fsmn_vad_instance = fsmnvad.FSMNVad()

    print("Models loaded.")
    print("Start plotting...")

    for wav_idx, wav_path in enumerate(wav_list):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_VAD_compare.png")

        print(f"[{wav_idx + 1}/{len(wav_list)}] Processing: {wav_path}")

        # -------------------------
        # Silero VAD
        # -------------------------
        silero_probs, silero_hop = silero_vad_process_wav(
            silero_model,
            wav_path
        )

        # -------------------------
        # TEN VAD
        # -------------------------
        ten_probs, ten_hop = ten_vad_process_wav(
            wav_path,
            hop_size=ten_hop_size,
            threshold=ten_threshold
        )

        # -------------------------
        # FSMN VAD
        # -------------------------
        fsmn_probs, fsmn_hop = fsmn_vad_process_wav(
            fsmn_vad_instance,
            wav_path
        )

        # -------------------------
        # 绘图并保存
        # -------------------------
        plot_vad_comparison_for_one_file(
            wav_path=wav_path,
            save_path=save_path,
            silero_probs=silero_probs,
            silero_hop=silero_hop,
            ten_probs=ten_probs,
            ten_hop=ten_hop,
            fsmn_probs=fsmn_probs,
            fsmn_hop=fsmn_hop,
            silero_threshold=silero_threshold,
            ten_threshold=ten_threshold,
            fsmn_threshold=fsmn_threshold,
        )

        print(f"Saved: {save_path}")

    print("All VAD comparison figures saved.")