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

class GeneralDatasetPathNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "数据集文件夹路径"
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_dataset_path"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_dataset_path(self, dataset_path):
        normalized_path = normalize_wsl_path(dataset_path)
        
        if not os.path.exists(normalized_path):
            print(f"警告: 路径不存在: {normalized_path}")
        
        return (dataset_path,)

class ArBucketsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ar_buckets": ("STRING", {
                    "default": "[[512, 512], [448, 576]]",
                    "multiline": False,
                    "tooltip": "宽高对分桶配置，可选，不能与宽高比分桶同时使用"
                }),
            }
        }
    
    RETURN_TYPES = ("ar_buckets",)
    RETURN_NAMES = ("ar_buckets",)
    FUNCTION = "process_ar_buckets"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def process_ar_buckets(self, ar_buckets):
        try:
            ar_buckets_str = ar_buckets.strip()
            
            if not ar_buckets_str:
                print("警告: 宽高比分桶配置为空，使用默认值")
                return ("[[512, 512], [448, 576]]",)
            
            print(f"宽高比分桶配置: {ar_buckets_str}")
            return (ar_buckets_str,)
            
        except Exception as e:
            print(f"处理宽高比分桶配置时出错: {str(e)}")
            print("使用默认宽高比分桶配置: [[512, 512], [448, 576]]")
            return ("[[512, 512], [448, 576]]",)

class EditModelDatasetPathNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "生成图像路径 - 模型要学习生成的图像"
                }),
                "control_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "原图像路径 - 与目标图像对应的控制图像"
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_edit_dataset_paths"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_edit_dataset_paths(self, target_path, control_path):
        normalized_target_path = normalize_wsl_path(target_path)
        normalized_control_path = normalize_wsl_path(control_path)
        
        if not os.path.exists(normalized_target_path):
            print(f"警告: 目标路径不存在: {normalized_target_path}")
        
        if not os.path.exists(normalized_control_path):
            print(f"警告: 控制路径不存在: {normalized_control_path}")
        
        dataset_config = {
            "path": target_path,
            "control_path": control_path
        }
        
        return (dataset_config,)

        
class MultiImageEditDatasetPathNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "control_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_multi_edit_dataset_paths"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_multi_edit_dataset_paths(self, target_path, control_paths):
        normalized_target_path = normalize_wsl_path(target_path)
        
        if not os.path.exists(normalized_target_path):
            print(f"警告: 目标路径不存在: {normalized_target_path}")
        
        paths = [p.strip() for p in control_paths.split('\n') if p.strip()]
        
        for idx, cp in enumerate(paths):
            normalized_cp = normalize_wsl_path(cp)
            if not os.path.exists(normalized_cp):
                print(f"警告: 控制路径 {idx+1} 不存在: {normalized_cp}")
        
        dataset_config = {
            "path": target_path,
            "control_paths": paths
        }
        
        return (dataset_config,)



class MaskDatasetPathNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "训练图像路径 - 包含训练图片的文件夹"
                }),
                "mask_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "遮罩图像路径 - 白色训练，黑色遮蔽，灰色部分训练。遮罩文件名需与训练图片匹配"
                }),
            }
        }
    
    RETURN_TYPES = ("input_path",)
    RETURN_NAMES = ("input_path",)
    FUNCTION = "get_mask_dataset_paths"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def get_mask_dataset_paths(self, dataset_path, mask_path):
        normalized_dataset_path = normalize_wsl_path(dataset_path)
        normalized_mask_path = normalize_wsl_path(mask_path)
        
        if not os.path.exists(normalized_dataset_path):
            print(f"警告: 数据集路径不存在: {normalized_dataset_path}")
        
        if not os.path.exists(normalized_mask_path):
            print(f"警告: 遮罩路径不存在: {normalized_mask_path}")
        
        dataset_config = {
            "path": dataset_path,
            "mask_path": mask_path
        }
        
        return (dataset_config,)


class FrameBucketsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_buckets": ("STRING", {
                    "default": "[1, 33, 81, 97]",
                    "multiline": False,
                    "tooltip": "帧数分桶配置，专用于视频模型训练，格式：[1, 33, 81, 97]"
                }),
            }
        }
    
    RETURN_TYPES = ("frame_buckets",)
    RETURN_NAMES = ("frame_buckets",)
    FUNCTION = "process_frame_buckets"
    CATEGORY = "Diffusion-Pipe/dataset"
    
    def process_frame_buckets(self, frame_buckets):
        try:
            frame_buckets_str = frame_buckets.strip()
            
       
            if not frame_buckets_str:
                print("警告: 帧桶配置为空，使用默认值")
                return ("[1, 33, 65, 97]",)
            
            print(f"帧桶配置: {frame_buckets_str}")
            return (frame_buckets_str,)
            
        except Exception as e:
            print(f"处理帧桶配置时出错: {str(e)}")
            print("使用默认帧桶配置: [1, 33, 65, 97]")
            return ("[1, 33, 65, 97]",)
