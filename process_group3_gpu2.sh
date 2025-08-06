#!/bin/bash

# 🎯 批量处理 Opens2S 数据 - GPU 2 组
# 处理情感、通用问答和语言相关的jsonl文件

echo "🚀 开始处理第3组数据 (GPU 2) - 情感、通用问答和语言相关文件..."

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=2

# 基础配置
SCRIPT_PATH="./construct_opens2s_data.py"
INPUT_DIR="/share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny"
OUTPUT_DIR="./open_s2s_ultravoice"
PREFIX_PATH="/share/nlp/tuwenming/projects/UltraVoice_dev/data"
TOKENIZER_PATH="/share/nlp/tuwenming/models/zai-org/glm-4-voice-tokenizer"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 第3组文件列表 - 情感、通用问答和语言相关
FILES=(
    "ultravoice_emotion_sad_5497.jsonl"
    "ultravoice_emotion_surprised_5492.jsonl"
    "ultravoice_generalqa_en_40000.jsonl"
    "ultravoice_language_chinese_5998.jsonl"
    "ultravoice_language_japanese_5994.jsonl"
    "ultravoice_language_korean_5990.jsonl"
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
    
    echo "⚡ 正在处理: $file (GPU 2)"
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

echo "🎉 第3组数据处理完成 (GPU 2)!"
echo "📁 输出目录: $OUTPUT_DIR"