import json
import os

def normalize_wsl_path(path):
    if not path:
        return path
        
    if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
        drive_letter = path[0].lower()
        rest_path = path[3:].replace('\\', '/')
        
        if drive_letter == 'z':
            if rest_path.startswith('home/'):
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        else:
            return f'/mnt/{drive_letter}/{rest_path}'
    
    elif path.startswith('/'):
        return path.replace('\\', '/')
    
    else:
        return path


class EvalDatasetConfig:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("input_path", {
                    "tooltip": "评估数据集输入路径，必选，根据不同训练目的，选择不同节点"
                }),
                
                "resolutions": ("STRING", {
                    "default": "[512]",
                    "multiline": False,
                    "tooltip": "评估分辨率，可以是单个数值（正方形）或 [宽度, 高度] 对,例如: [1280, 720]"
                }),
                
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
                
                "num_repeats": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "评估数据集重复次数"
                }),
            },
            "optional": {
                "frame_buckets": ("frame_buckets",{
                "tooltip": "帧分桶设置，例如: [1, 33] 或 [1, 33, 65, 97]，专用与视频模型训练"                            
            }),
                "ar_buckets": ("ar_buckets",{
                    "tooltip": "宽高比分桶设置，例如:[[512, 512], [448, 576]]",
            }),            
            }
        }
    
    RETURN_TYPES = ("EVAL_DATASET_CONFIG",)
    RETURN_NAMES = ("eval_dataset_config",)
    FUNCTION = "generate_config"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def generate_config(self, input_path, resolutions, enable_ar_bucket, min_ar, max_ar, 
                       num_ar_buckets, num_repeats, frame_buckets=None, ar_buckets=None):
        try:
            dataset_path = None
            control_path = None
            is_edit_model = False
            
            if isinstance(input_path, dict):
                dataset_path = input_path.get("path")
                control_path = input_path.get("control_path")
                is_edit_model = True
            elif isinstance(input_path, str):
                dataset_path = input_path
            
            if is_edit_model and frame_buckets is not None and frame_buckets.strip():
                raise ValueError(
                    "error，you can't use frame_buckets and edit_model at the same time"
                 
                )
            
            resolutions_list = self._parse_list_input(resolutions, "resolutions")
            
            frame_buckets_list = None
            if frame_buckets is not None and frame_buckets.strip():
                frame_buckets_list = self._parse_list_input(frame_buckets, "frame_buckets")
            
            ar_buckets_list = None
            if ar_buckets is not None and ar_buckets.strip():
                ar_buckets_list = self._parse_list_input(ar_buckets, "ar_buckets")
            
            config_lines = []
            
            if len(resolutions_list) == 1 and isinstance(resolutions_list[0], (int, float)):
                config_lines.append(f"resolutions = [{int(resolutions_list[0])}]")
            else:
                config_lines.append(f"resolutions = {resolutions_list}")
            
            config_lines.append(f"enable_ar_bucket = {str(enable_ar_bucket).lower()}")
            
            if enable_ar_bucket:
                config_lines.extend([
                    f"min_ar = {min_ar}",
                    f"max_ar = {max_ar}",
                    f"num_ar_buckets = {num_ar_buckets}",
                ])
            
            if ar_buckets_list is not None:
                config_lines.append(f"ar_buckets = {ar_buckets_list}")
            
            if frame_buckets_list is not None:
                config_lines.append(f"frame_buckets = {frame_buckets_list}")
            
            if control_path:

                normalized_dataset_path = normalize_wsl_path(dataset_path) if dataset_path else "C:\\path\\to\\target\\images"
                normalized_control_path = normalize_wsl_path(control_path) if control_path else "C:\\path\\to\\control\\images"
                config_lines.extend([
                    "[[directory]]",
                    f"path = '{normalized_dataset_path}'",
                    f"control_path = '{normalized_control_path}'",
                    f"num_repeats = {num_repeats}",
                ])
            else:
                normalized_dataset_path = normalize_wsl_path(dataset_path) if dataset_path else "C:\\path\\to\\your\\eval\\dataset"
                config_lines.extend([
                    "[[directory]]",
                    f"path = '{normalized_dataset_path}'",
                    f"num_repeats = {num_repeats}",
                ])
            
            config_content = "\n".join(config_lines)
            
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_path = os.path.join(current_dir, "..", "dataset", "evaldataset.toml")
                normalized_output_path = os.path.normpath(output_path)
                
                output_dir = os.path.dirname(normalized_output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                import datetime
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_cwd = os.getcwd()
                print("\n" + "="*80)
                print(f"[评估数据集配置] 配置文件已保存到: {normalized_output_path}")
                print(f"[评估数据集配置] 生成时间: {current_time}")
                print(f"[评估数据集配置] 当前工作目录: {current_cwd}")
                print("="*80 + "\n")
                
                with open(normalized_output_path, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                
                display_path = normalized_output_path.replace('\\', '/')
                print("="*80)
                print(f"评估数据集配置已保存到: {display_path}")
                print("="*80)
                
                return (config_content,)
            except Exception as e:
                print(f"保存评估配置文件失败: {str(e)}")
                return (config_content,)
            
        except Exception as e:
            error_msg = f"生成评估数据集配置失败: {str(e)}"
            return (error_msg,)
    
    def _parse_list_input(self, input_str, param_name):
        try:
            input_str = input_str.strip()
            
            if not input_str:
                raise ValueError(f"{param_name} 不能为空")
            
            if input_str.startswith('[') and input_str.endswith(']'):
                parsed = json.loads(input_str)
                if isinstance(parsed, list):
                    return parsed
                else:
                    raise ValueError("必须是列表格式")
            else:
                try:
                    value = float(input_str)
                    return [int(value) if value.is_integer() else value]
                except:
                    raise ValueError("格式错误，应为数字或 [数字1, 数字2, ...] 格式")
                    
        except json.JSONDecodeError:
            raise ValueError(f"{param_name} JSON 格式错误")
        except Exception as e:
            raise ValueError(f"{param_name} 解析失败: {str(e)}") 
    


