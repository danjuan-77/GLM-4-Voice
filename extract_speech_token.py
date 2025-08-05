#!/usr/bin/env python3
"""
音频token提取脚本
从音频文件中提取speech token的独立脚本
"""

import os
import sys
import argparse

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
    
    print(f"正在处理音频文件: {audio_path}")
    
    # 使用utils中的extract_speech_token函数
    audio_tokens = extract_speech_token(
        whisper_model, feature_extractor, [audio_path]
    )[0]
    
    if len(audio_tokens) == 0:
        print("警告: 没有提取到音频token")
        return []
    
    print(f"成功提取 {len(audio_tokens)} 个音频token")
    return audio_tokens


def format_tokens(audio_tokens: list, include_special_tokens: bool = True):
    """格式化token输出"""
    if not audio_tokens:
        return ""
    
    if include_special_tokens:
        # 只输出音频token格式，不包含begin/end标记
        formatted_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        return formatted_tokens
    else:
        # 只返回原始token数字
        return audio_tokens


def main():
    parser = argparse.ArgumentParser(description="从音频文件中提取speech token")
    parser.add_argument("audio_path", type=str, help="输入音频文件路径")
    parser.add_argument("--tokenizer-path", type=str, 
                       default="/home/tuwenming/Models/zai-org/glm-4-voice-tokenizer",
                       help="语音tokenizer模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="使用的设备")
    parser.add_argument("--output", type=str, help="输出文件路径（可选）")
    parser.add_argument("--format", type=str, default="special",
                       choices=["special", "raw"],
                       help="输出格式: special包含特殊token, raw只输出数字")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = "cpu"
    
    try:
        # 加载模型
        whisper_model, feature_extractor = load_models(args.tokenizer_path, args.device)
        
        # 提取token
        audio_tokens = extract_tokens_from_audio(args.audio_path, whisper_model, feature_extractor)
        
        # 格式化输出
        include_special = (args.format == "special")
        formatted_output = format_tokens(audio_tokens, include_special)
        
        # 输出结果
        if args.output:
            # 保存到文件
            with open(args.output, 'w', encoding='utf-8') as f:
                if include_special:
                    f.write(formatted_output)
                else:
                    f.write(','.join(map(str, audio_tokens)))
            print(f"结果已保存到: {args.output}")
        else:
            # 输出到控制台
            if args.verbose:
                print("\n=== 提取结果 ===")
                print(f"Token数量: {len(audio_tokens)}")
                if include_special:
                    print(f"格式化输出:\n{formatted_output}")
                else:
                    print(f"原始token: {audio_tokens}")
            else:
                if include_special:
                    print(formatted_output)
                else:
                    print(','.join(map(str, audio_tokens)))
                    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
