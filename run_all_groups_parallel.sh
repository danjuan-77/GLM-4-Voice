#!/bin/bash

# 🚀 并行运行所有GPU组处理脚本
# 同时启动4个GPU组的数据处理任务

echo "🎯 开始并行处理所有Opens2S数据组..."
echo "使用4个GPU同时处理24个jsonl文件"
echo "=========================================="

# 检查脚本文件是否存在
scripts=(
    "process_group1_gpu0.sh"
    "process_group2_gpu1.sh"
    "process_group3_gpu2.sh"
    "process_group4_gpu3.sh"
)

for script in "${scripts[@]}"; do
    if [ ! -f "$script" ]; then
        echo "❌ 脚本文件不存在: $script"
        exit 1
    fi
    chmod +x "$script"
done

echo "✅ 所有脚本文件检查完成，开始并行执行..."
echo ""

# 并行启动所有脚本
echo "🚀 启动GPU 0组 (口音相关)..."
./process_group1_gpu0.sh > logs_gpu0.log 2>&1 &
PID1=$!

echo "🚀 启动GPU 1组 (描述和情感相关)..."
./process_group2_gpu1.sh > logs_gpu1.log 2>&1 &
PID2=$!

echo "🚀 启动GPU 2组 (情感、通用问答和语言相关)..."
./process_group3_gpu2.sh > logs_gpu2.log 2>&1 &
PID3=$!

echo "🚀 启动GPU 3组 (语速和音量相关)..."
./process_group4_gpu3.sh > logs_gpu3.log 2>&1 &
PID4=$!

echo ""
echo "📊 所有任务已启动，进程ID："
echo "  GPU 0 组: PID $PID1"
echo "  GPU 1 组: PID $PID2"
echo "  GPU 2 组: PID $PID3"
echo "  GPU 3 组: PID $PID4"
echo ""
echo "📝 日志文件："
echo "  GPU 0: logs_gpu0.log"
echo "  GPU 1: logs_gpu1.log"
echo "  GPU 2: logs_gpu2.log"
echo "  GPU 3: logs_gpu3.log"
echo ""

# 等待所有任务完成
echo "⏳ 等待所有任务完成..."
wait $PID1
exit_code1=$?

wait $PID2
exit_code2=$?

wait $PID3
exit_code3=$?

wait $PID4
exit_code4=$?

echo ""
echo "🏁 所有任务执行完成！"
echo "=========================================="
echo "📊 执行结果："
echo "  GPU 0 组: $([ $exit_code1 -eq 0 ] && echo "✅ 成功" || echo "❌ 失败 (退出码: $exit_code1)")"
echo "  GPU 1 组: $([ $exit_code2 -eq 0 ] && echo "✅ 成功" || echo "❌ 失败 (退出码: $exit_code2)")"
echo "  GPU 2 组: $([ $exit_code3 -eq 0 ] && echo "✅ 成功" || echo "❌ 失败 (退出码: $exit_code3)")"
echo "  GPU 3 组: $([ $exit_code4 -eq 0 ] && echo "✅ 成功" || echo "❌ 失败 (退出码: $exit_code4)")"
echo ""

# 检查输出目录
if [ -d "./open_s2s_ultravoice" ]; then
    echo "📁 输出文件统计："
    echo "  输出目录: ./open_s2s_ultravoice/"
    file_count=$(find ./open_s2s_ultravoice -name "*.jsonl" | wc -l)
    echo "  生成文件数: $file_count"
    echo ""
    echo "📋 生成的文件列表："
    ls -la ./open_s2s_ultravoice/*.jsonl 2>/dev/null | head -10
    if [ $file_count -gt 10 ]; then
        echo "  ... (还有 $((file_count - 10)) 个文件)"
    fi
fi

echo ""
echo "🎉 批量处理完成！"
echo "🔍 如需查看详细日志，请检查 logs_gpu*.log 文件"