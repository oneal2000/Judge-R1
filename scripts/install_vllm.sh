#!/bin/bash
# ===========================================
# vLLM 安装脚本（适配 MS-Swift 3.9.3 + CUDA 12.8）
#
# 安装前请确保：
# 1. 已激活 swift 环境: conda activate swift
# 2. 备份当前环境: conda env export > swift_backup.yml
#
# 安装后验证：
# 1. python -c "import vllm; print(vllm.__version__)"
# 2. python -c "import swift; print(swift.__version__)"
# ===========================================

set -e

echo "=========================================="
echo "  vLLM 安装脚本"
echo "  目标: MS-Swift 3.9.3 + vLLM 0.11.0"
echo "=========================================="

# 检查是否在正确的 conda 环境中
if [[ "$CONDA_DEFAULT_ENV" != "swift" ]]; then
    echo "[ERROR] 请先激活 swift 环境: conda activate swift"
    exit 1
fi

# Step 0: 备份当前环境
echo ""
echo "[Step 0] 备份当前环境..."
BACKUP_FILE="/data-share/chenxuanyi/internship/JuDGE_RL/swift_env_backup_$(date +%Y%m%d_%H%M%S).yml"
conda env export > "${BACKUP_FILE}"
echo "  备份保存到: ${BACKUP_FILE}"

# Step 1: 记录当前版本
echo ""
echo "[Step 1] 当前环境版本:"
echo "  ms-swift: $(pip show ms-swift 2>/dev/null | grep Version | awk '{print $2}')"
echo "  torch: $(pip show torch 2>/dev/null | grep Version | awk '{print $2}')"
echo "  transformers: $(pip show transformers 2>/dev/null | grep Version | awk '{print $2}')"
echo "  vllm: $(pip show vllm 2>/dev/null | grep Version | awk '{print $2}' || echo 'not installed')"

# Step 2: 升级 ms-swift 到 3.9.3
echo ""
echo "[Step 2] 升级 ms-swift 到 3.9.3..."
pip install ms-swift==3.9.3

# Step 3: 安装 vLLM 0.11.0（适配 CUDA 12.8）
echo ""
echo "[Step 3] 安装 vLLM 0.11.0..."
# 使用 CUDA 12.8 的 PyTorch 索引
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

# Step 4: 验证安装
echo ""
echo "[Step 4] 验证安装..."
python -c "
import torch
import transformers
import swift
import vllm

print('========== 安装验证 ==========')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
print(f'MS-Swift: {swift.__version__}')
print(f'vLLM: {vllm.__version__}')
print('==============================')
"

# Step 5: 测试 GRPO 导入
echo ""
echo "[Step 5] 测试 GRPO 模块导入..."
python -c "
from swift.llm.train.rlhf import rlhf_main
from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
print('GRPO 模块导入成功!')
"

echo ""
echo "=========================================="
echo "✅ vLLM 安装完成!"
echo "=========================================="
echo ""
echo "如果出现问题，可以回滚:"
echo "  conda env remove -n swift"
echo "  conda env create -f ${BACKUP_FILE}"
echo ""
echo "下一步: 运行训练脚本时使用 vLLM 加速"
echo "  VLLM_MODE=colocate bash bash/agent/train_agent_rl_lawselect.sh"
