import json
import os

def normalize_wsl_path(path):
    """
    规范化WSL2环境下的路径
    将Windows路径转换为WSL路径，或保持Linux路径不变
    """
    if not path:
        return path
        
    # 如果是Windows驱动器路径格式 (如 Z:\path 或 C:\path)
    if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
        drive_letter = path[0].lower()
        rest_path = path[3:].replace('\\', '/')
        
        # 特殊处理Z:驱动器（通常映射到WSL根目录）
        if drive_letter == 'z':
            # 如果rest_path以home开头，直接映射到根目录
            if rest_path.startswith('home/'):
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        else:
            return f'/mnt/{drive_letter}/{rest_path}'
    
    # 如果已经是Linux路径格式，保持正斜杠
    elif path.startswith('/'):
        return path.replace('\\', '/')
    
    # 相对路径处理 - 保持原样，不进行路径规范化（避免在Windows下转换为反斜杠）
    else:
        return path

class GeneralDatasetConfig:
    """
    通用数据集配置节点
    用于配置训练数据集的各种参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入路径
                "input_path": ("input_path", {
                    "tooltip": "数据集输入路径，必选，根据不同训练目的，选择不同节点"
                }),
                
                # 基础分辨率设置
                "resolutions": ("STRING", {
                    "default": "[512]",
                    "multiline": False,
                    "tooltip": "训练分辨率，可以是单个数值（正方形）或 [宽度, 高度] 对,例如: [1280, 720]"
                }),
                
                # 宽高比分桶设置
                "enable_ar_bucket": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用宽高比分桶设置"
                    }),
                    
                "min_ar": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "最小宽高比"
                }),
                "max_ar": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "最大宽高比"
                }),
                "num_ar_buckets": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "宽高比分桶数量"
                }),
                
                # 数据集重复次数
                "num_repeats": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数据集重复次数，用于增加训练数据的有效使用次数"
                }),
                
                # 输出路径设置
                "output_path": ("STRING", {
                    "default": "D:/ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI/dataset/dataset.toml",
                    "multiline": False,
                    "tooltip": "配置文件保存路径"
                }),
            },
            "optional": {
                # 帧分桶设置（视频训练） 
                "frame_buckets": ("frame_buckets",{
                "tooltip": "帧分桶设置，例如: [1, 33] 或 [1, 33, 65, 97]，专用与视频模型训练"                            
            }),
                "ar_buckets": ("ar_buckets",{
                    "tooltip": "宽高比分桶设置，例如:[[512, 512], [448, 576]]",
            }),            
            }
        }
    
    RETURN_TYPES = ("DATASET_CONFIG",)
    RETURN_NAMES = ("dataset_config",)
    FUNCTION = "generate_config"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def generate_config(self, input_path, resolutions, enable_ar_bucket, min_ar, max_ar, 
                       num_ar_buckets, num_repeats, output_path, frame_buckets=None, ar_buckets=None):
        """
        生成数据集配置文件内容
        """
        try:
            # 处理input_path参数
            dataset_path = None
            control_path = None
            is_edit_model = False
            
            if isinstance(input_path, dict):
                # 如果是字典（来自EditModelDatasetPathNode）
                dataset_path = input_path.get("path")
                control_path = input_path.get("control_path")
                is_edit_model = True
            elif isinstance(input_path, str):
                # 如果是字符串（来自GeneralDatasetPathNode）
                dataset_path = input_path
            
            # 验证模型类型兼容性
            if is_edit_model and frame_buckets is not None and frame_buckets.strip():
                raise ValueError(
                    "error，you can't use frame_buckets and edit_model at the same time"
                 
                )
            
            # 解析分辨率设置
            resolutions_list = self._parse_list_input(resolutions, "resolutions")
            
            # 解析帧分桶设置（如果提供）
            frame_buckets_list = None
            if frame_buckets is not None and frame_buckets.strip():
                frame_buckets_list = self._parse_list_input(frame_buckets, "frame_buckets")
            
            # 解析宽高比分桶设置（如果提供）
            ar_buckets_list = None
            if ar_buckets is not None and ar_buckets.strip():
                ar_buckets_list = self._parse_list_input(ar_buckets, "ar_buckets")
            
            # 构建配置内容
            config_lines = []
            
            # 分辨率配置
            if len(resolutions_list) == 1 and isinstance(resolutions_list[0], (int, float)):
                config_lines.append(f"resolutions = [{int(resolutions_list[0])}]")
            else:
                config_lines.append(f"resolutions = {resolutions_list}")
            
            # 宽高比分桶配置
            config_lines.append(f"enable_ar_bucket = {str(enable_ar_bucket).lower()}")
            
            if enable_ar_bucket:
                config_lines.extend([
                    f"min_ar = {min_ar}",
                    f"max_ar = {max_ar}",
                    f"num_ar_buckets = {num_ar_buckets}",
                ])
            
            # 宽高比分桶配置（如果提供）
            if ar_buckets_list is not None:
                config_lines.append(f"ar_buckets = {ar_buckets_list}")
            
            # 帧分桶配置（仅在提供时添加）
            if frame_buckets_list is not None:
                config_lines.append(f"frame_buckets = {frame_buckets_list}")
            
            # 添加目录配置
            if control_path:
                # 编辑模型配置（有control_path）
                # 使用WSL路径规范化处理路径
                normalized_dataset_path = normalize_wsl_path(dataset_path) if dataset_path else "/path/to/target/images"
                normalized_control_path = normalize_wsl_path(control_path) if control_path else "/path/to/control/images"
                config_lines.extend([
                    f"path = '{normalized_dataset_path}'",
                    f"control_path = '{normalized_control_path}'",
                ])
            else:
                # 通用数据集配置
                # 使用WSL路径规范化处理路径
                normalized_dataset_path = normalize_wsl_path(dataset_path) if dataset_path else "/path/to/your/dataset"
                config_lines.extend([
                    "[[directory]]",
                    f"path = '{normalized_dataset_path}'",
                    f"num_repeats = {num_repeats}",
                ])
            
            config_content = "\n".join(config_lines)
            
            # 保存配置文件
            if output_path:
                try:
                    # WSL2环境路径处理：将Windows路径转换为WSL路径
                    normalized_output_path = normalize_wsl_path(output_path)
                    
                    # 确保输出目录存在
                    output_dir = os.path.dirname(normalized_output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # 在配置内容前添加路径注释和时间戳
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_cwd = os.getcwd()
                    # 输出配置信息到控制台
                    print(f"[数据集配置] 配置文件已保存到: {normalized_output_path}")
                    print(f"[数据集配置] 生成时间: {current_time}")
                    print(f"[数据集配置] 当前工作目录: {current_cwd}")
                    
                    config_with_path = config_content
                    
                    with open(normalized_output_path, 'w', encoding='utf-8') as f:
                        f.write(config_with_path)
                    
                    # 显示保存路径时使用正斜杠（便于阅读）
                    display_path = normalized_output_path.replace('\\', '/')
                    print(f"数据集配置已保存到: {display_path}")
                    
                    # 返回包含路径注释的配置
                    return (config_with_path,)
                except Exception as e:
                    print(f"保存配置文件失败: {str(e)}")
            else:
                print("未指定输出路径，仅生成配置内容")
            
            return (config_content,)
            
        except Exception as e:
            error_msg = f"生成数据集配置失败: {str(e)}"
            return (error_msg,)
    
    def _parse_list_input(self, input_str, param_name):
        """
        解析列表输入字符串
        """
        try:
            # 移除空白字符
            input_str = input_str.strip()
            
            if not input_str:
                raise ValueError(f"{param_name} 不能为空")
            
            # 尝试解析为 JSON
            if input_str.startswith('[') and input_str.endswith(']'):
                parsed = json.loads(input_str)
                if isinstance(parsed, list):
                    return parsed
                else:
                    raise ValueError("必须是列表格式")
            else:
                # 尝试解析为单个数值
                try:
                    value = float(input_str)
                    return [int(value) if value.is_integer() else value]
                except:
                    raise ValueError("格式错误，应为数字或 [数字1, 数字2, ...] 格式")
                    
        except json.JSONDecodeError:
            raise ValueError(f"{param_name} JSON 格式错误")
        except Exception as e:
            raise ValueError(f"{param_name} 解析失败: {str(e)}") 
    
