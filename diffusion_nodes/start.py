import os
import sys
import subprocess
import threading
from datetime import datetime
import toml
import tempfile
import json
import time
import signal
import queue
from pathlib import Path

# 尝试导入配置解析器
try:
    # 相对导入 (在 ComfyUI 中作为包导入时)
    from ..utils.config_parser import ConfigParser
except ImportError:
    # 绝对导入 (直接运行时)
    import os
    import sys
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        from utils.config_parser import ConfigParser
    except ImportError:
        # 如果找不到配置解析器，创建一个简单的替代
        class ConfigParser:
            @staticmethod
            def merge_configs(dataset_config, train_config):
                return {**dataset_config, **train_config}

class Train:
    def __init__(self):
        self.training_process = None
        self.log_queue = queue.Queue()
        self.is_training = False
        # 注册全局实例
        try:
            from .train_monitor import set_global_train_instance
            set_global_train_instance(self)
        except ImportError:
            pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("DATASET_CONFIG", {
                    "tooltip": "数据集配置（来自GeneralDatasetConfig节点）"
                }),
                "train_config": ("TRAIN_CONFIG", {
                    "tooltip": "训练配置（来自GeneralConfig节点）"
                }),
                "config_path": ("config_path", {
                    "tooltip": "配置文件路径（来自GeneralConfig节点）"
                }),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "log_output")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Train"
    
    def execute(self, dataset_config, train_config, config_path):
        """ComfyUI节点的执行入口"""
        return self.start_training(dataset_config, train_config, config_path)
    
    def normalize_wsl_path(self, path):
        """规范化WSL2环境下的路径"""
        if not path:
            return path
            
        # 如果是Windows驱动器路径格式
        if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
            drive_letter = path[0].lower()
            rest_path = path[3:].replace('\\', '/')
            
            # 特殊处理Z:驱动器 - 通常映射到WSL根目录
            if drive_letter == 'z':
                # Z盘通常映射到WSL的根目录，直接返回Linux路径
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        
        # 如果已经是Linux路径格式
        elif path.startswith('/'):
            return path.replace('\\', '/')
        
        # 相对路径处理
        else:
            return path

    def create_config_files(self, dataset_config, train_config):
        """创建临时配置文件 - 已弃用，现在使用 config_save_path"""
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="diffusion_train_")
            
            # 检查并处理dataset_config类型
            if isinstance(dataset_config, str):
                # 如果是字符串，尝试解析为JSON或TOML
                try:
                    dataset_config = json.loads(dataset_config)
                except json.JSONDecodeError:
                    try:
                        dataset_config = toml.loads(dataset_config)
                    except Exception:
                        # 如果都解析失败，创建一个基本的配置
                        dataset_config = {"path": dataset_config}
            elif not isinstance(dataset_config, dict):
                return None, f"dataset_config类型不正确: {type(dataset_config)}, 应该是字典或字符串"
            
            # 检查并处理train_config类型
            if isinstance(train_config, str):
                # 如果是字符串，尝试解析为JSON或TOML
                try:
                    train_config = json.loads(train_config)
                except json.JSONDecodeError:
                    try:
                        train_config = toml.loads(train_config)
                    except Exception:
                        return None, f"无法解析train_config字符串: {train_config}"
            elif not isinstance(train_config, dict):
                return None, f"train_config类型不正确: {type(train_config)}, 应该是字典或字符串"
            
            # 创建数据集配置文件
            dataset_config_path = os.path.join(temp_dir, "dataset_config.toml")
            with open(dataset_config_path, 'w', encoding='utf-8') as f:
                # 规范化路径
                normalized_dataset_config = {}
                for key, value in dataset_config.items():
                    if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                        normalized_dataset_config[key] = self.normalize_wsl_path(value)
                    else:
                        normalized_dataset_config[key] = value
                toml.dump(normalized_dataset_config, f)
            
            # 创建训练配置文件
            train_config_copy = train_config.copy()
            
            # 规范化训练配置中的路径
            if 'output_dir' in train_config_copy:
                train_config_copy['output_dir'] = self.normalize_wsl_path(train_config_copy['output_dir'])
            
            # 设置数据集配置路径
            train_config_copy['dataset'] = dataset_config_path
            
            train_config_path = os.path.join(temp_dir, "train_config.toml")
            with open(train_config_path, 'w', encoding='utf-8') as f:
                toml.dump(train_config_copy, f)
            
            return train_config_path, temp_dir
            
        except Exception as e:
            return None, str(e)

    def log_reader(self, process, log_queue):
        """读取训练进程的输出日志"""
        try:
            for line in iter(process.stdout.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    log_queue.put(decoded_line)
                    print(f"[Training] {decoded_line}")
            
            for line in iter(process.stderr.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    log_queue.put(f"ERROR: {decoded_line}")
                    print(f"[Training Error] {decoded_line}")
                    
        except Exception as e:
            log_queue.put(f"Log reader error: {str(e)}")

    def start_training(self, dataset_config, train_config, config_path):
        """启动训练进程"""
        try:
            # 检查是否已经在训练
            if self.is_training and self.training_process and self.training_process.poll() is None:
                return "ALREADY_RUNNING", "训练已在进行中，请等待当前训练完成"
            
            # 预处理train_config，确保它是字典类型
            if isinstance(train_config, str):
                try:
                    # 尝试解析为JSON
                    import json
                    train_config = json.loads(train_config)
                except json.JSONDecodeError:
                    try:
                        # 尝试解析为TOML
                        import toml
                        train_config = toml.loads(train_config)
                    except:
                        # 如果都失败，创建基础配置
                        train_config = {}
            
            # 确保train_config是字典类型
            if not isinstance(train_config, dict):
                train_config = {}
            
            # 检查配置文件路径
            if not config_path:
                return "ERROR", "未指定配置文件保存路径 (config_path)"
            
            # 规范化配置文件路径
            config_path = self.normalize_wsl_path(config_path)
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                return "ERROR", f"配置文件不存在: {config_path}"
            
            # 获取训练脚本路径
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_script = os.path.join(current_dir, "train.py")
            
            if not os.path.exists(train_script):
                return "ERROR", f"找不到训练脚本: {train_script}"
            
            # 构建训练命令 - 使用 DeepSpeed 启动
            num_gpus = train_config.get('number_of_gpus', 1)
            
            cmd = [
                "deepspeed",
                f"--num_gpus={num_gpus}",
                train_script,
                "--deepspeed",
                "--config", config_path
            ]
            
            # 如果配置中指定了其他参数，添加到命令中
            if train_config.get('regenerate_cache', False):
                cmd.append("--regenerate_cache")
            
            if train_config.get('trust_cache', False):
                cmd.append("--trust_cache")
                
            if 'master_port' in train_config:
                cmd.extend(["--master_port", str(train_config['master_port'])])
            
            # 设置环境变量
            env = os.environ.copy()
            
            # 设置 NCCL 环境变量以避免通信问题
            env['NCCL_P2P_DISABLE'] = "1"
            env['NCCL_IB_DISABLE'] = "1"
            
            # 如果是多GPU训练，设置相关环境变量
            if num_gpus > 1:
                env['WORLD_SIZE'] = str(num_gpus)
                env['RANK'] = '0'
                env['LOCAL_RANK'] = '0'
                env['MASTER_ADDR'] = 'localhost'
                env['MASTER_PORT'] = str(train_config.get('master_port', 29500))
            
            # 启动训练进程
            print(f"启动训练命令: {' '.join(cmd)}")
            print(f"使用配置文件: {config_path}")
            print(f"GPU数量: {num_gpus}")
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=1,
                universal_newlines=False
            )
            
            # 启动日志读取线程
            log_thread = threading.Thread(
                target=self.log_reader,
                args=(self.training_process, self.log_queue),
                daemon=True
            )
            log_thread.start()
            
            self.is_training = True
            
            # 等待一小段时间检查进程是否正常启动
            time.sleep(2)
            
            if self.training_process.poll() is not None:
                # 进程已经结束，可能是启动失败
                return_code = self.training_process.returncode
                error_msg = f"训练进程启动失败，返回码: {return_code}"
                
                # 尝试读取错误信息
                try:
                    stderr_output = self.training_process.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        error_msg += f"\n错误信息: {stderr_output}"
                except:
                    pass
                
                self.is_training = False
                return "ERROR", error_msg
            
            # 收集初始日志
            initial_logs = []
            log_timeout = time.time() + 5  # 5秒超时
            
            while time.time() < log_timeout:
                try:
                    log_line = self.log_queue.get_nowait()
                    initial_logs.append(log_line)
                except queue.Empty:
                    time.sleep(0.1)
                    continue
            
            log_output = "\n".join(initial_logs) if initial_logs else "训练已启动，正在初始化..."
            
            return "TRAINING_STARTED", f"训练成功启动!\nPID: {self.training_process.pid}\n配置文件: {config_path}\n\n初始日志:\n{log_output}"
            
        except Exception as e:
            self.is_training = False
            error_msg = f"启动训练时发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return "ERROR", error_msg
    
    def stop_training(self):
        """停止训练进程"""
        if self.training_process and self.training_process.poll() is None:
            try:
                # 尝试优雅地终止进程
                self.training_process.terminate()
                
                # 等待进程结束，最多等待10秒
                try:
                    self.training_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # 如果进程没有在10秒内结束，强制杀死
                    self.training_process.kill()
                    self.training_process.wait()
                
                self.is_training = False
                return "STOPPED", "训练已停止"
                
            except Exception as e:
                return "ERROR", f"停止训练时发生错误: {str(e)}"
        else:
            return "NOT_RUNNING", "没有正在运行的训练进程"
    
    def get_training_status(self):
        """获取训练状态"""
        if not self.training_process:
            return "NOT_STARTED", "训练未启动"
        
        if self.training_process.poll() is None:
            # 进程仍在运行
            logs = []
            try:
                while True:
                    log_line = self.log_queue.get_nowait()
                    logs.append(log_line)
            except queue.Empty:
                pass
            
            log_output = "\n".join(logs[-50:]) if logs else "训练进行中..."  # 只显示最近50行日志
            return "RUNNING", f"训练正在进行中 (PID: {self.training_process.pid})\n\n最新日志:\n{log_output}"
        else:
            # 进程已结束
            return_code = self.training_process.returncode
            self.is_training = False
            
            if return_code == 0:
                return "COMPLETED", f"训练已完成 (返回码: {return_code})"
            else:
                return "FAILED", f"训练失败 (返回码: {return_code})"
    
