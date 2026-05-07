#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#

import sys
import os
import json
import numpy as np
import scipy.io.wavfile as Wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad


def prob_to_segments(prob_list, threshold, hop_size, sample_rate):

    frame_shift_ms = hop_size * 1000.0 / sample_rate

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

            # 转换为秒（这里除以1000）
            start_s = (start_frame * frame_shift_ms) / 1000.0
            end_s = (end_frame * frame_shift_ms) / 1000.0

            segments.append([start_s, end_s])
            in_speech = False

    if in_speech:
        end_frame = len(binary)

        start_s = (start_frame * frame_shift_ms) / 1000.0
        end_s = (end_frame * frame_shift_ms) / 1000.0

        segments.append([start_s, end_s])

    return segments, binary


if __name__ == "__main__":

    input_dir = "your_path"
    output_txt = "your_path"
    fail_txt = "your_path"

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    hop_size = 256       # 16 ms per frame at 16 kHz
    threshold = 0.5
    
    total_audio_count = 0
    success_audio_count = 0
    failed_audio_files = []

    # 以写入模式打开文件，用于记录所有音频的 VAD 检测片段
    with open(output_txt, "w", encoding="utf-8") as out_f:
        # 遍历输入目录下的所有子目录和文件
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    total_audio_count += 1
                    input_file = os.path.join(root, file)
                    abs_input_file = os.path.abspath(input_file)

                    try:
                        sr, data = Wavfile.read(abs_input_file)
                    except Exception as e:
                        print(f"Error reading {abs_input_file}: {e}")
                        failed_audio_files.append(abs_input_file)
                        continue

                    # TODO: 若存在有些采样率非 16000 且不强制跳过，可根据需求加重采样
                    if sr != 16000:
                        print(f"Skipping {file}: TEN VAD only supports 16kHz, but got {sr}")
                        failed_audio_files.append(abs_input_file)
                        continue

                    try:
                        if data.ndim > 1:
                            data = data[:, 0]
    
                        if data.dtype != np.int16:
                            if np.issubdtype(data.dtype, np.floating):
                                data = np.clip(data, -1.0, 1.0)
                                data = (data * 32767).astype(np.int16)
                            else:
                                data = data.astype(np.int16)
    
                        # 每次必须实例化一个新的对象来保持独立的 VAD 状态
                        ten_vad_instance = TenVad(hop_size, threshold)
    
                        num_frames = data.shape[0] // hop_size
                        prob_list = []
    
                        for i in range(num_frames):
                            audio_data = data[i * hop_size: (i + 1) * hop_size]
                            out_probability, _ = ten_vad_instance.process(audio_data)
                            prob_list.append(float(out_probability))
    
                        del ten_vad_instance
    
                        # 解析出分段
                        segments, _ = prob_to_segments(
                            prob_list=prob_list,
                            threshold=threshold,
                            hop_size=hop_size,
                            sample_rate=sr
                        )
    
                        # 格式化输出为: 音频文件名字 (开始, 结束), (开始, 结束) ...
                        file_no_ext = os.path.splitext(file)[0]
                        if len(segments) > 0:
                            # 格式化为保留 3 位小数的秒数
                            seg_strs = [f"({s:.3f}, {e:.3f})" for s, e in segments]
                            out_f.write(f"{file_no_ext} {', '.join(seg_strs)}\n")
                        else:
                            out_f.write(f"{file_no_ext}\n")
                            
                        success_audio_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {abs_input_file}: {e}")
                        failed_audio_files.append(abs_input_file)
                        continue
                        
    # 将失败的音频路径写入 fail.txt
    if failed_audio_files:
        with open(fail_txt, "w", encoding="utf-8") as f_fail:
            for fail_file in failed_audio_files:
                f_fail.write(fail_file + "\n")

    print("\n" + "="*40)
    print(f"总音频数: {total_audio_count}")
    print(f"成功处理音频数: {success_audio_count}")
    print(f"处理失败音频数: {len(failed_audio_files)}")
    print("="*40)
    print(f"检测完成，成功结果已保存至 {output_txt}")
    if failed_audio_files:
        print(f"失败的文件绝对路径已保存至 {fail_txt}")