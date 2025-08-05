#!/usr/bin/env python3
"""
批量构建Opens2S数据集
从原始jsonl数据集转换为Opens2S格式，并提取speech token
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token


def load_models(tokenizer_path: str, device: str = "cuda"):
    """加载语音tokenizer模型"""
    print(f"正在加载模型从: {tokenizer_path}")
    
    # 加载Whisper VQ编码器
    whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    print("WhisperVQEncoder 加载完成")
    
    # 加载特征提取器
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    print("WhisperFeatureExtractor 加载完成")
    
    return whisper_model, feature_extractor


def extract_tokens_from_audio(audio_path: str, whisper_model, feature_extractor):
    """从音频文件中提取speech token"""
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 使用utils中的extract_speech_token函数
    audio_tokens = extract_speech_token(
        whisper_model, feature_extractor, [audio_path]
    )[0]
    
    return audio_tokens


def format_speech_units(audio_tokens: list):
    """格式化speech units为<|audio_0|><|audio_1|>格式"""
    if not audio_tokens:
        return ""
    
    return "".join([f"<|audio_{x}|>" for x in audio_tokens])


def convert_data_item(item: dict, prefix_path: str, whisper_model, feature_extractor):
    """转换单条数据为Opens2S格式"""
    
    # 构建绝对路径
    instruction_wav_path = os.path.join(prefix_path, item["instruction_wav_path"])
    response_wav_path = os.path.join(prefix_path, item["response_wav_path"])
    
    # 检查文件是否存在
    if not os.path.exists(instruction_wav_path):
        raise FileNotFoundError(f"指令音频文件不存在: {instruction_wav_path}")
    if not os.path.exists(response_wav_path):
        raise FileNotFoundError(f"回复音频文件不存在: {response_wav_path}")
    
    # 提取回复音频的speech token
    try:
        response_tokens = extract_tokens_from_audio(response_wav_path, whisper_model, feature_extractor)
        speech_units = format_speech_units(response_tokens)
    except Exception as e:
        print(f"警告: 提取token失败 {response_wav_path}: {e}")
        speech_units = ""
    
    # 构建Opens2S格式
    converted_item = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "",
                        "audio": instruction_wav_path,
                        "speech_units": "",
                        "spk_emb": ""
                    }
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {
                        "text": item["response_text"],
                        "audio": "",
                        "speech_units": speech_units,
                        "spk_emb": ""
                    }
                ]
            }
        ]
    }
    
    return converted_item


def main():
    parser = argparse.ArgumentParser(description="批量构建Opens2S数据集")
    parser.add_argument("input_jsonl", type=str, help="输入jsonl文件路径")
    parser.add_argument("output_dir", type=str, help="输出目录路径")
    parser.add_argument("--prefix-path", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/data",
                       help="音频文件的前缀路径")
    parser.add_argument("--tokenizer-path", type=str, 
                       default="/share/nlp/tuwenming/models/zai-org/glm-4-voice-tokenizer",
                       help="语音tokenizer模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="使用的设备")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批处理大小（暂时只支持1）")
    parser.add_argument("--skip-errors", action="store_true",
                       help="跳过错误的数据项继续处理")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = "cpu"
    
    # 检查输入文件
    if not os.path.exists(args.input_jsonl):
        print(f"错误: 输入文件不存在: {args.input_jsonl}")
        sys.exit(1)
    
    # 检查前缀路径
    if not os.path.exists(args.prefix_path):
        print(f"错误: 前缀路径不存在: {args.prefix_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    input_basename = os.path.splitext(os.path.basename(args.input_jsonl))[0]
    output_filename = f"{input_basename}_opens2s_train.jsonl"
    output_jsonl = os.path.join(args.output_dir, output_filename)
    
    print(f"输出文件将保存为: {output_jsonl}")
    
    try:
        # 加载模型
        whisper_model, feature_extractor = load_models(args.tokenizer_path, args.device)
        
        # 读取输入数据
        print(f"正在读取输入文件: {args.input_jsonl}")
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            input_data = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"共读取到 {len(input_data)} 条数据")
        
        # 批量处理
        converted_data = []
        error_count = 0
        
        for i, item in enumerate(tqdm(input_data, desc="处理数据")):
            try:
                converted_item = convert_data_item(item, args.prefix_path, whisper_model, feature_extractor)
                converted_data.append(converted_item)
                
                if args.verbose and (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(input_data)} 条数据")
                    
            except Exception as e:
                error_count += 1
                if args.skip_errors:
                    print(f"跳过错误数据 {i}: {e}")
                    continue
                else:
                    print(f"处理数据 {i} 时出错: {e}")
                    sys.exit(1)
        
        # 保存结果
        print(f"正在保存结果到: {output_jsonl}")
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("转换完成!")
        print(f"成功处理: {len(converted_data)} 条数据")
        if error_count > 0:
            print(f"跳过错误: {error_count} 条数据")
        print(f"结果已保存到: {output_jsonl}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
