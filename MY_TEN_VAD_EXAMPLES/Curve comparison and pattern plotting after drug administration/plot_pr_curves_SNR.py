#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#  Refer to the "LICENSE" file in the root directory for more information.
#

import os
import sys
import glob
from pathlib import Path

import numpy as np
import torchaudio
import scipy.io.wavfile as Wavfile
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


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
# 2. 标签处理函数
# =========================

def convert_label_to_framewise(label_file, hop_size, sample_rate=16000):
    """
    将 .scv 标注文件转换成帧级标签。

    label 格式假设为：
    xxx,start,end,label,start,end,label,...

    label:
        0 = non-speech
        1 = speech
    """
    frame_duration = hop_size / sample_rate

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = lines[0].strip().split(",")[1:]

    start = np.array(content[::3], dtype=float)
    end = np.array(content[1:][::3], dtype=float)
    lab_manual = np.array(content[2:][::3], dtype=int)

    assert (
        len(start) == len(end)
        and len(start) == len(lab_manual)
        and len(end) == len(lab_manual)
    ), f"Label file format error: {label_file}"

    num = np.array(np.round((end - start) / frame_duration), dtype=np.int32)

    label_framewise = np.array([], dtype=np.float32)

    for segment_idx in range(len(num)):
        cur_lab = int(lab_manual[segment_idx])
        num_segment = int(num[segment_idx])

        if cur_lab == 1:
            segment_label = np.ones(num_segment, dtype=np.float32)
        else:
            segment_label = np.zeros(num_segment, dtype=np.float32)

        label_framewise = np.append(label_framewise, segment_label)

    frame_num = min(
        len(label_framewise),
        int((end[-1] - start[0]) / frame_duration)
    )

    label_framewise = label_framewise[:frame_num]

    return label_framewise


# =========================
# 3. 指标计算函数
# =========================

def get_precision_recall(vad_result, label, threshold):
    """
    根据阈值计算 Precision / Recall / FPR / FNR。
    """
    vad_result_hard = np.where(vad_result >= threshold, 1, 0)

    # labels=[0, 1] 可以避免某些极端情况下 confusion_matrix 不是 2x2
    TN, FP, FN, TP = confusion_matrix(
        label,
        vad_result_hard,
        labels=[0, 1]
    ).ravel()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall, FPR, FNR


# =========================
# 4. TEN VAD 推理
# =========================

def ten_vad_process_wav(ten_vad_instance, wav_path, hop_size=256):
    """
    TEN VAD 输出逐帧 voice probability。
    """
    sample_rate, data = Wavfile.read(wav_path)

    assert sample_rate == 16000, f"TEN VAD only supports 16kHz, but got {sample_rate}: {wav_path}"

    # 如果是双声道，取第一个声道
    if len(data.shape) > 1:
        data = data[:, 0]

    num_frames = data.shape[0] // hop_size
    voice_prob_arr = np.array([], dtype=np.float32)

    for i in range(num_frames):
        input_data = data[i * hop_size: (i + 1) * hop_size]
        voice_prob, _ = ten_vad_instance.process(input_data)
        voice_prob_arr = np.append(voice_prob_arr, voice_prob)

    return voice_prob_arr


# =========================
# 5. Silero VAD 推理
# =========================

def silero_vad_inference_single_file(model, wav_path):
    """
    Silero VAD 输出逐帧 speech probability。
    默认 window_size_samples = 512。
    """
    window_size_samples = 512
    speech_probs = np.array([], dtype=np.float32)

    wav, sr = torchaudio.load(wav_path)

    assert sr == 16000, f"Silero VAD only supports 16kHz in this script, but got {sr}: {wav_path}"

    # 如果是双声道，取第一个声道
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

def fsmn_vad_process_wav(fsmn_vad_instance, wav_path):
    """
    FSMN VAD 无后处理版本：
    直接取模型 raw scores，不经过 infer_offline / VAD 状态机 / 端点检测。
    """

    if not hasattr(fsmn_vad_instance, "raw_scores_offline"):
        raise AttributeError(
            "当前 FSMNVad 没有 raw_scores_offline() 接口。"
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
        step = int(round(wav_len / len(speech_scores)))
    else:
        step = int(sample_rate * 0.01)

    return speech_scores, step


# =========================
# 6. 保存 PR 数据
# =========================

def save_pr_data(save_path, pr_data):
    """
    保存格式：
        threshold precision recall
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for ind in range(pr_data.shape[0]):
            precision = pr_data[ind, 0]
            recall = pr_data[ind, 1]
            threshold = pr_data[ind, 2]
            f.write(f"{threshold:.2f} {precision:.4f} {recall:.4f}\n")


# =========================
# 7. 处理单个SNR数据集
# =========================

def process_snr_dataset(snr_value, snr_dir, label_dir, output_dir, 
                       silero_model, fsmn_vad_instance, script_dir):
    """
    处理单个SNR数据集，生成PR曲线和数据文件。
    """
    
    print(f"\n{'='*80}")
    print(f"Processing SNR: {snr_value} dB")
    print(f"Data directory: {snr_dir}")
    print(f"{'='*80}")
    
    sample_rate = 16000
    ten_hop_size = 256
    ten_threshold = 0.5
    
    # 使用列表替代np.array以提高追加效率
    label_ten_all = []
    vad_result_ten_all = []
    label_silero_all = []
    vad_result_silero_all = []
    label_fsmn_all = []
    vad_result_fsmn_all = []

    if not Path(snr_dir).exists():
        print(f"WARNING: SNR directory not found: {snr_dir}")
        return None
    
    wav_list = sorted([str(p) for p in Path(snr_dir).glob("*.wav")])
    
    if len(wav_list) == 0:
        print(f"WARNING: No wav files found in: {snr_dir}")
        return None
    
    print(f"Found {len(wav_list)} wav files.")
    
    for wav_idx, wav_path in enumerate(wav_list):
        wav_filename = os.path.basename(wav_path)
        print(f"  [{wav_idx + 1}/{len(wav_list)}] Processing: {wav_filename}", end="\r")
        
        label_filename = wav_filename.replace(".wav", ".txt")
        label_file = os.path.join(label_dir, label_filename)
        
        if not os.path.exists(label_file):
            continue
        
        try:
            ten_vad_instance = TenVad(ten_hop_size, ten_threshold)
            label_ten = convert_label_to_framewise(label_file, hop_size=ten_hop_size, sample_rate=sample_rate)
            vad_result_ten = ten_vad_process_wav(ten_vad_instance, wav_path, hop_size=ten_hop_size)
            frame_num_ten = min(len(label_ten), len(vad_result_ten))
            if frame_num_ten > 1:
                vad_result_ten_all.extend(vad_result_ten[1:frame_num_ten])
                label_ten_all.extend(label_ten[:frame_num_ten - 1])
            del ten_vad_instance
        except Exception as e:
            print(f"    WARNING: TEN VAD failed for {wav_filename}: {e}")
        
        try:
            vad_result_silero, silero_hop_size = silero_vad_inference_single_file(silero_model, wav_path)
            label_silero = convert_label_to_framewise(label_file, hop_size=silero_hop_size, sample_rate=sample_rate)
            frame_num_silero = min(len(label_silero), len(vad_result_silero))
            vad_result_silero_all.extend(vad_result_silero[:frame_num_silero])
            label_silero_all.extend(label_silero[:frame_num_silero])
        except Exception as e:
            print(f"    WARNING: Silero VAD failed for {wav_filename}: {e}")
        
        try:
            vad_result_fsmn, fsmn_hop_size = fsmn_vad_process_wav(fsmn_vad_instance, wav_path)
            label_fsmn = convert_label_to_framewise(label_file, hop_size=fsmn_hop_size, sample_rate=sample_rate)
            frame_num_fsmn = min(len(label_fsmn), len(vad_result_fsmn))
            vad_result_fsmn_all.extend(vad_result_fsmn[:frame_num_fsmn])
            label_fsmn_all.extend(label_fsmn[:frame_num_fsmn])
        except Exception as e:
            print(f"    WARNING: FSMN VAD failed for {wav_filename}: {e}")
    
    # 转换列表为numpy数组
    vad_result_ten_all = np.array(vad_result_ten_all, dtype=np.float32)
    label_ten_all = np.array(label_ten_all, dtype=np.float32)
    vad_result_silero_all = np.array(vad_result_silero_all, dtype=np.float32)
    label_silero_all = np.array(label_silero_all, dtype=np.float32)
    vad_result_fsmn_all = np.array(vad_result_fsmn_all, dtype=np.float32)
    label_fsmn_all = np.array(label_fsmn_all, dtype=np.float32)
    
    print(f"\nAll files processed for SNR {snr_value} dB.")
    print(f"TEN frames:    pred={len(vad_result_ten_all)}, label={len(label_ten_all)}")
    print(f"Silero frames: pred={len(vad_result_silero_all)}, label={len(label_silero_all)}")
    print(f"FSMN frames:   pred={len(vad_result_fsmn_all)}, label={len(label_fsmn_all)}")
    
    if (len(vad_result_ten_all) == 0 or len(label_ten_all) == 0):
        print(f"WARNING: No valid data for SNR {snr_value} dB, skipping...")
        return None
    
    threshold_arr = np.arange(0, 1.01, 0.01)
    pr_data_ten = np.zeros((len(threshold_arr), 3), dtype=np.float32)
    pr_data_silero = np.zeros((len(threshold_arr), 3), dtype=np.float32)
    pr_data_fsmn = np.zeros((len(threshold_arr), 3), dtype=np.float32)
    
    print("Computing Precision-Recall data...")
    
    for ind, threshold in enumerate(threshold_arr):
        precision_ten, recall_ten, FPR_ten, FNR_ten = get_precision_recall(vad_result_ten_all, label_ten_all, threshold)
        pr_data_ten[ind] = precision_ten, recall_ten, threshold
        
        if len(vad_result_silero_all) > 0 and len(label_silero_all) > 0:
            precision_silero, recall_silero, FPR_silero, FNR_silero = get_precision_recall(vad_result_silero_all, label_silero_all, threshold)
            pr_data_silero[ind] = precision_silero, recall_silero, threshold
        
        if len(vad_result_fsmn_all) > 0 and len(label_fsmn_all) > 0:
            precision_fsmn, recall_fsmn, FPR_fsmn, FNR_fsmn = get_precision_recall(vad_result_fsmn_all, label_fsmn_all, threshold)
            pr_data_fsmn[ind] = precision_fsmn, recall_fsmn, threshold
    
    print("Plotting PR Curve...")
    plt.figure(figsize=(10, 7))
    pr_data_ten_to_plot = pr_data_ten[:-1]
    pr_data_silero_to_plot = pr_data_silero[:-1]
    pr_data_fsmn_to_plot = pr_data_fsmn[:-1]
    
    plt.plot(pr_data_ten_to_plot[:, 1], pr_data_ten_to_plot[:, 0], color="red", label="TEN VAD", linewidth=2.5)
    
    if len(vad_result_silero_all) > 0:
        plt.plot(pr_data_silero_to_plot[:, 1], pr_data_silero_to_plot[:, 0], color="blue", label="Silero VAD", linewidth=2.5)
    
    if len(vad_result_fsmn_all) > 0:
        plt.plot(pr_data_fsmn_to_plot[:, 1], pr_data_fsmn_to_plot[:, 0], color="green", label="FSMN VAD", linewidth=2.5)
    
    plt.xlabel("Recall", fontsize=18, fontweight="bold", color="black")
    plt.ylabel("Precision", fontsize=18, fontweight="bold", color="black")
    
    snr_str = f"{snr_value}"
    plt.title(f"PR Curves - clean_100_free-sound ({snr_str} dB)", fontsize=20, color="black", fontweight="bold")
    plt.grid(True)
    plt.xlim(0.65, 1.00)
    plt.ylim(0.70, 1.00)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    legend = plt.legend(fontsize=16)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    
    os.makedirs(output_dir, exist_ok=True)
    image_filename = f"clean_100_free-sound_{snr_str}dB.png"
    save_path = os.path.join(output_dir, image_filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"PR Curve image saved: {save_path}")
    plt.close()
    
    pr_data_ten_save_path = os.path.join(output_dir, f"PR_data_TEN_VAD_{snr_str}dB.txt")
    pr_data_silero_save_path = os.path.join(output_dir, f"PR_data_SILERO_VAD_{snr_str}dB.txt")
    pr_data_fsmn_save_path = os.path.join(output_dir, f"PR_data_FSMN_VAD_{snr_str}dB.txt")
    
    save_pr_data(pr_data_ten_save_path, pr_data_ten)
    if len(vad_result_silero_all) > 0:
        save_pr_data(pr_data_silero_save_path, pr_data_silero)
    if len(vad_result_fsmn_all) > 0:
        save_pr_data(pr_data_fsmn_save_path, pr_data_fsmn)
    
    print(f"TEN VAD PR data saved: {pr_data_ten_save_path}")
    
    return save_path


# =========================
# 8. 主程序
# =========================

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ===== 多SNR配置 =====
    dataset_base_dir = Path(r"D:\VAD\1.vad_learn\data_processing\train_clean_100_free-sound")
    label_dir = Path(r"D:\VAD\1.vad_learn\data_processing\dataset_label")
    output_save_dir = Path(r"D:\VAD\1.vad_learn\ten-vad\SNR_PR_save")
    snr_values = [-10, -5, 0, 5, 10]
    silero_model_path = r"D:/VAD/1.vad_learn/silero-vad/src/silero_vad/data/silero_vad.jit"
    
    if not dataset_base_dir.exists():
        raise FileNotFoundError(f"Dataset base directory not found: {dataset_base_dir}")
    
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    print("="*80)
    print("Multi-SNR PR Curves Generator")
    print("="*80)
    print(f"Dataset base directory: {dataset_base_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Output directory: {output_save_dir}")
    print(f"Processing SNRs: {snr_values}")
    print("="*80)
    
    os.makedirs(output_save_dir, exist_ok=True)
    
    print("\nLoading models...")
    silero_model = init_jit_model(silero_model_path)
    fsmn_vad_instance = fsmnvad.FSMNVad()
    print("Models loaded.")
    
    results = []
    for snr_value in snr_values:
        # 构造实际的SNR目录名（格式: snr_-10dB, snr_0dB, snr_5dB等）
        snr_dir_name = f"snr_{snr_value}dB"
        snr_data_dir = dataset_base_dir / snr_dir_name
        
        result = process_snr_dataset(
            snr_value=snr_value,
            snr_dir=str(snr_data_dir),
            label_dir=str(label_dir),
            output_dir=str(output_save_dir),
            silero_model=silero_model,
            fsmn_vad_instance=fsmn_vad_instance,
            script_dir=script_dir
        )
        
        if result is not None:
            results.append((snr_value, result))
    
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Successfully processed {len(results)} SNR datasets:")
    for snr_val, img_path in results:
        snr_str = f"{snr_val}"
        print(f"  - SNR {snr_str} dB: {img_path}")
    print("="*80)
    print("All processing completed!")