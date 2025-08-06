#!/bin/bash

# 🎯 批量处理 Opens2S 数据 - GPU 0 组
# 处理口音相关的jsonl文件

echo "🚀 开始处理第1组数据 (GPU 0) - 口音相关文件..."

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 基础配置
SCRIPT_PATH="./construct_opens2s_data.py"
INPUT_DIR="/share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny"
OUTPUT_DIR="./open_s2s_ultravoice"
PREFIX_PATH="/share/nlp/tuwenming/projects/UltraVoice_dev/data"
TOKENIZER_PATH="/share/nlp/tuwenming/models/zai-org/glm-4-voice-tokenizer"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 第1组文件列表 - 口音相关
FILES=(
    "ultravoice_accent_AU_5990.jsonl"
    "ultravoice_accent_CA_5988.jsonl"
    "ultravoice_accent_GB_5998.jsonl"
    "ultravoice_accent_IN_5996.jsonl"
    "ultravoice_accent_SG_5999.jsonl"
    "ultravoice_accent_ZA_5996.jsonl"
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
    
    echo "⚡ 正在处理: $file (GPU 0)"
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

echo "🎉 第1组数据处理完成 (GPU 0)!"
echo "📁 输出目录: $OUTPUT_DIR"