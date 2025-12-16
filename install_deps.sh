#!/bin/bash
# 依赖安装脚本
# 使用方式: bash install_deps.sh [方式]
#   方式1: 完整安装（默认）
#   方式2: 核心依赖最小化安装
#   方式3: 使用conda安装（推荐，避免编译）

set -e  # 遇到错误立即退出

echo "======================================"
echo "新闻推荐系统 - 依赖安装脚本"
echo "======================================"
echo ""

# 检查conda环境
if ! conda info --envs | grep -q "ds-lab2"; then
    echo "错误：未找到 ds-lab2 conda环境"
    echo "请先创建环境: conda create -n ds-lab2 python=3.8"
    exit 1
fi

# 激活环境
echo "激活 ds-lab2 环境..."
eval "$(conda shell.bash hook)"
conda activate ds-lab2

# 检查Python版本
PYTHON_VERSION=$(python --version | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"
echo ""

# 获取安装方式
METHOD=${1:-1}

case $METHOD in
    1)
        echo "【方式1】完整安装（pip）"
        echo "包含核心依赖 + faiss-cpu + 可视化库"
        echo ""
        pip install pandas==1.5.3 numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 lightgbm==3.3.5 tqdm==4.65.0
        echo ""
        echo "安装可选依赖..."
        pip install faiss-cpu==1.7.4 || echo "⚠️  faiss-cpu安装失败，将使用numpy降级方案"
        pip install matplotlib==3.7.1 seaborn==0.12.2 || echo "⚠️  可视化库安装失败（不影响核心功能）"
        ;;

    2)
        echo "【方式2】最小化安装（仅核心依赖）"
        echo "只安装运行baseline和main必需的包"
        echo ""
        pip install pandas==1.5.3 numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 lightgbm==3.3.5 tqdm==4.65.0
        ;;

    3)
        echo "【方式3】使用conda安装（推荐）"
        echo "从conda-forge安装预编译的二进制包，避免编译错误"
        echo ""
        conda install -y pandas=1.5.3 numpy=1.23.5 scipy=1.10.1 scikit-learn=1.2.2 lightgbm=3.3.5 tqdm=4.65.0 -c conda-forge
        echo ""
        echo "安装可选依赖..."
        conda install -y faiss-cpu=1.7.4 -c conda-forge || echo "⚠️  faiss-cpu安装失败，将使用numpy降级方案"
        conda install -y matplotlib=3.7.1 seaborn=0.12.2 -c conda-forge || echo "⚠️  可视化库安装失败（不影响核心功能）"
        ;;

    *)
        echo "错误：未知的安装方式 '$METHOD'"
        echo "用法: bash install_deps.sh [1|2|3]"
        echo "  1 - 完整安装（pip）"
        echo "  2 - 最小化安装（仅核心依赖）"
        echo "  3 - conda安装（推荐）"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "安装完成！"
echo "======================================"
echo ""
echo "验证安装："
python -c "import pandas, numpy, sklearn, lightgbm, tqdm; print('✅ 核心依赖安装成功')"

# 检查可选依赖
python -c "import faiss; print('✅ faiss-cpu 可用')" 2>/dev/null || echo "⚠️  faiss-cpu 未安装（将使用numpy降级）"
python -c "import matplotlib, seaborn; print('✅ 可视化库可用')" 2>/dev/null || echo "⚠️  可视化库未安装（不影响核心功能）"

echo ""
echo "下一步："
echo "  cd src"
echo "  python baseline.py  # 运行baseline快速测试"
