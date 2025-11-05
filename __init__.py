"""
Diffusion-Pipe ComfyUI 自定义节点包
为 ComfyUI 提供 Diffusion-Pipe 功能集成
"""

# 导入主要的节点注册
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 导出给 ComfyUI 使用
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("\n" + "=" * 60)
print("Diffusion_pipe_in_ComfyUI Plugin Loaded")
print("=" * 60)
print(f"🎉Nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print("\nAvailable Nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  • {display_name}")
print("=" * 60 + "\n")

 
 