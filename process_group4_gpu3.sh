#!/bin/bash

# 🎯 批量处理 Opens2S 数据 - GPU 3 组
# 处理语速和音量相关的jsonl文件

echo "🚀 开始处理第4组数据 (GPU 3) - 语速和音量相关文件..."

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=3

# 基础配置
SCRIPT_PATH="./construct_opens2s_data.py"
INPUT_DIR="/share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny"
OUTPUT_DIR="./open_s2s_ultravoice"
PREFIX_PATH="/share/nlp/tuwenming/projects/UltraVoice_dev/data"
TOKENIZER_PATH="/share/nlp/tuwenming/models/zai-org/glm-4-voice-tokenizer"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 第4组文件列表 - 语速和音量相关
FILES=(
    "ultravoice_speed_fast_5492.jsonl"
    "ultravoice_speed_normal_5492.jsonl"
    "ultravoice_speed_slow_5499.jsonl"
    "ultravoice_volume_high_5491.jsonl"
    "ultravoice_volume_low_5492.jsonl"
    "ultravoice_volume_normal_5490.jsonl"
)

echo "📋 当前处理文件列表:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done
echo ""

# 处理每个文件
for file in "${FILES[@]}"; do
    input_file="$INPUT_DIR/$file"
    
    if [ ! -f "$input_file" ]; then
        echo "❌ 文件不存在: $input_file"
        continue
    fi
    
    echo "⚡ 正在处理: $file (GPU 3)"
    python3 "$SCRIPT_PATH" \
        "$input_file" \
        "$OUTPUT_DIR" \
        --prefix-path "$PREFIX_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --device cuda \
        --skip-errors \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ 完成处理: $file"
    else
        echo "❌ 处理失败: $file"
    fi
    echo "----------------------------------------"
done

echo "🎉 第4组数据处理完成 (GPU 3)!"
echo "📁 输出目录: $OUTPUT_DIR"