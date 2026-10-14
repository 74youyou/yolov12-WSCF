# -*- coding: utf-8 -*-
"""
环境自检脚本：检测 flash-attn 是否真正可用
"""
import torch
import platform

print("=" * 60)
print("1. 基础环境信息")
print(f"   Python 版本: {platform.python_version()}")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   CUDA 版本 (编译时): {torch.version.cuda}")
print(f"   PyTorch 是否开启 CUDA: {torch.cuda.is_available()}")
print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
print(f"   GPU 算力 (Capability): {torch.cuda.get_device_capability(0)}")  # 4090 应该是 (8, 9)

print("\n2. 尝试导入 flash_attn 模块")
try:
    import flash_attn

    print(f"   flash_attn 模块导入成功！版本号: {flash_attn.__version__}")
except Exception as e:
    print(f"   ❌ 模块导入失败！报错信息: {e}")
    print("   (这说明你虽然 pip 安装了，但 Python 无法加载它，可能是路径或编译问题)")
    exit()

print("\n3. 尝试运行前向传播（真实测试底层内核）")
try:
    # 创建一个简单的注意力输入
    # batch_size=2, seq_len=1024, head_dim=128 (常规大小)
    q = torch.randn(2, 16, 1024, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(2, 16, 1024, 128, device='cuda', dtype=torch.float16)
    v = torch.randn(2, 16, 1024, 128, device='cuda', dtype=torch.float16)

    from flash_attn import flash_attn_func

    # 运行测试
    out = flash_attn_func(q, k, v)
    print(f"   ✅ Flash Attention 内核运行成功！输出形状: {out.shape}")

except Exception as e:
    print(f"   ❌ 内核运行失败！这就是你之前看到那个报错的真正原因：")
    print(f"   {type(e).__name__}: {e}")
    print("\n   ---- 排查建议 ----")
    print("   1. 报错中如果包含 'undefined symbol' 或 'no kernel image'：说明编译版本不匹配。")
    print("   2. 报错中如果包含 'CUDA error: no kernel image is available for execution on the device'：")
    print("      说明你当前的 flash-attn 只编译了 sm_80 (A100) 的内核，没有编译 sm_89 (4090) 的内核。")
    print("   3. 报错中如果包含 'ImportError'：说明你的 Python 虚拟环境乱了。")