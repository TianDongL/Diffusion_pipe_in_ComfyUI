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

try:
    from ..utils.config_parser import ConfigParser
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        from utils.config_parser import ConfigParser
    except ImportError:
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
            "optional": {
                "resume_from_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "从指定检查点继续训练，例如：'20250212_07-06-40' 或留空表示不从检查点恢复"
                }),
                "reset_dataloader": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "重置数据加载器状态（在从检查点恢复时使用）"
                }),
                "regenerate_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "强制重新生成缓存文件"
                }),
                "cache_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅生成缓存然后退出，不进行训练"
                }),
                "trust_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "信任现有缓存，不进行验证"
                }),
                "i_know_what_i_am_doing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "跳过某些安全检查（仅在你知道自己在做什么时使用）"
                }),
                "dump_dataset": ("STRING", {
                    "default": "",
                    "tooltip": "导出数据集到指定路径（用于调试）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "log_output")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Train"
    
    def execute(self, dataset_config, train_config, config_path, 
                resume_from_checkpoint="", reset_dataloader=False, 
                regenerate_cache=False, cache_only=False, 
                trust_cache=False, i_know_what_i_am_doing=False,
                dump_dataset=""):
        """ComfyUI节点的执行入口"""
        return self.start_training(
            dataset_config, train_config, config_path, 
            resume_from_checkpoint, reset_dataloader, 
            regenerate_cache, cache_only, trust_cache, 
            i_know_what_i_am_doing, dump_dataset
        )
    
    def normalize_wsl_path(self, path):
        """规范化WSL2环境下的路径"""
        if not path:
            return path
            
        # 如果是Windows驱动器路径格式
        if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
            drive_letter = path[0].lower()
            rest_path = path[3:].replace('\\', '/')
            
            if drive_letter == 'z':
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        
        elif path.startswith('/'):
            return path.replace('\\', '/')
        
        else:
            return path
    def log_reader(self, stream, log_queue, prefix="", stream_name="stream"):
        """改进的日志读取器，支持进度条显示"""
        try:
            buffer = ""
            last_was_progress = False
            
            while True:
                chunk = stream.read(256)
                if not chunk:
                    break
                    
                text = chunk.decode('utf-8', errors='ignore')
                
                for char in text:
                    if char == '\n':
                        if buffer.strip():
                            if last_was_progress:
                                print()
                            line = f"{prefix}{buffer}" if prefix else buffer
                            print(line)
                            log_queue.put(line)
                            last_was_progress = False
                        buffer = ""
                    elif char == '\r':
                        if buffer.strip():
                            # 检测进度条模式：包含百分比、进度条符号等
                            is_progress = '%|' in buffer or '|/' in buffer or ('[' in buffer and ']' in buffer)
                            if is_progress:
                                # 进度条：使用回车符在同一行更新
                                print(f"\r{buffer}", end='', flush=True)
                                last_was_progress = True
                            else:
                                # 普通行：正常打印
                                if last_was_progress:
                                    print()
                                line = f"{prefix}{buffer}" if prefix else buffer
                                print(line)
                                log_queue.put(line)
                                last_was_progress = False
                        buffer = ""
                    else:
                        buffer += char
                    
            # 处理剩余缓冲区内容
            if buffer.strip():
                if last_was_progress:
                    print()
                line = f"{prefix}{buffer}" if prefix else buffer
                print(line)
                log_queue.put(line)
        except Exception as e:
            error_msg = f"ERROR reading {stream_name}: {str(e)}"
            print(error_msg)
            log_queue.put(error_msg)

    def start_training(self, dataset_config, train_config, config_path, 
                      resume_from_checkpoint="", reset_dataloader=False, 
                      regenerate_cache=False, cache_only=False, 
                      trust_cache=False, i_know_what_i_am_doing=False,
                      dump_dataset=""):
        """启动训练进程"""
        try:
            if self.is_training and self.training_process and self.training_process.poll() is None:
                return "ALREADY_RUNNING", "训练已在进行中，请等待当前训练完成"
            
            if isinstance(train_config, str):
                try:
                    import json
                    train_config = json.loads(train_config)
                except json.JSONDecodeError:
                    try:
                        import toml
                        train_config = toml.loads(train_config)
                    except:
                        train_config = {}
            
            if not isinstance(train_config, dict):
                train_config = {}
            
            if not config_path:
                return "ERROR", "未指定配置文件保存路径 (config_path)"
            
            config_path = self.normalize_wsl_path(config_path)
            
            if not os.path.exists(config_path):
                return "ERROR", f"配置文件不存在: {config_path}"
            
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_script = os.path.join(current_dir, "train.py")
            
            if not os.path.exists(train_script):
                return "ERROR", f"找不到训练脚本: {train_script}"
            
            num_gpus = train_config.get('number_of_gpus', 1)
            
            cmd = [
                "deepspeed",
                f"--num_gpus={num_gpus}",
                train_script,
                "--deepspeed",
                "--config", config_path
            ]
            
            train_cmd_args = train_config.get('_train_cmd_args', {})
            
            # 1. 处理布尔型参数
            bool_params = {
                'reset_dataloader': reset_dataloader,
                'regenerate_cache': regenerate_cache,
                'cache_only': cache_only,
                'trust_cache': trust_cache,
                'i_know_what_i_am_doing': i_know_what_i_am_doing
            }
            
            for arg_name, node_value in bool_params.items():
                if node_value:  # 节点参数为True
                    cmd.append(f"--{arg_name}")
                elif arg_name in train_cmd_args and train_cmd_args.get(arg_name, False):
                    cmd.append(f"--{arg_name}")
                elif train_config.get(arg_name, False):
                    cmd.append(f"--{arg_name}")
            
            final_master_port = None
            if 'master_port' in train_cmd_args:
                final_master_port = train_cmd_args['master_port']
            elif 'master_port' in train_config:
                final_master_port = train_config['master_port']
            
            if final_master_port is not None:
                cmd.extend(["--master_port", str(final_master_port)])
            
            final_dump_dataset = dump_dataset  
            if 'dump_dataset' in train_cmd_args and train_cmd_args['dump_dataset']:
                final_dump_dataset = train_cmd_args['dump_dataset']
            elif 'dump_dataset' in train_config and train_config['dump_dataset']:
                final_dump_dataset = train_config['dump_dataset']
            
            if final_dump_dataset and final_dump_dataset.strip():
                cmd.extend(["--dump_dataset", final_dump_dataset.strip()])
            
            final_resume = resume_from_checkpoint 
            if 'resume_from_checkpoint' in train_cmd_args:
                resume_value = train_cmd_args['resume_from_checkpoint']
                if isinstance(resume_value, bool) and resume_value:
                    final_resume = "latest"  
                elif isinstance(resume_value, str):
                    final_resume = resume_value
            
            if final_resume and final_resume.strip():
                cmd.extend(["--resume_from_checkpoint", final_resume.strip()])
            
            # 设置环境变量
            env = os.environ.copy()
            
            # 设置 NCCL 环境变量以避免通信问题
            env['NCCL_P2P_DISABLE'] = "1"
            env['NCCL_IB_DISABLE'] = "1"
            
            # 禁用Python输出缓冲，确保日志实时输出
            env['PYTHONUNBUFFERED'] = "1"
            
            # 如果是多GPU训练，设置相关环境变量
            if num_gpus > 1:
                env['WORLD_SIZE'] = str(num_gpus)
                env['RANK'] = '0'
                env['LOCAL_RANK'] = '0'
                env['MASTER_ADDR'] = 'localhost'
                env['MASTER_PORT'] = str(train_config.get('master_port', 29500))
            
            # 启动训练进程
            print("\n" + "="*80)
            print("Starting Training Process")
            print("="*80)
            print(f"Command: {' '.join(cmd)}")
            print(f"Config: {config_path}")
            print(f"GPUs: {num_gpus}")
            if resume_from_checkpoint and resume_from_checkpoint.strip():
                print(f"Resume from: {resume_from_checkpoint.strip()}")
            print("="*80 + "\n")
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,  # 无缓冲模式，确保实时输出
                universal_newlines=False
            )
            
            # 启动日志读取线程 - 分别处理 stdout 和 stderr
            stdout_thread = threading.Thread(
                target=self.log_reader,
                args=(self.training_process.stdout, self.log_queue, "", "stdout"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self.log_reader,
                args=(self.training_process.stderr, self.log_queue, "Training ", "stderr"),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            self.is_training = True
            
            # 等待一小段时间检查进程是否正常启动
            time.sleep(2)
            
            if self.training_process.poll() is not None:
                # 进程已经结束，可能是启动失败
                return_code = self.training_process.returncode
                print("\n" + "="*80)
                print("Training Process Failed to Start")
                print("="*80)
                error_msg = f"Exit code: {return_code}"
                
                # 尝试读取错误信息
                try:
                    stderr_output = self.training_process.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        error_msg += f"\n{stderr_output}"
                        print(stderr_output, flush=True)
                except:
                    pass
                
                print("="*80 + "\n")
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
            
            resume_info = f"\n从检查点恢复: {resume_from_checkpoint.strip()}" if resume_from_checkpoint and resume_from_checkpoint.strip() else ""
            
            return "TRAINING_STARTED", f"训练成功启动!\nPID: {self.training_process.pid}\n配置文件: {config_path}{resume_info}\n\n初始日志:\n{log_output}"
            
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
                
                print("\n" + "="*80)
                print("Training Stopped by User")
                print("="*80 + "\n")
                
                self.is_training = False
                return "STOPPED", "Training stopped"
                
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
                print("\n" + "="*80)
                print("Training Completed Successfully")
                print("="*80 + "\n")
                return "COMPLETED", f"Training completed (Exit code: {return_code})"
            else:
                print("\n" + "="*80)
                print("Training Failed")
                print("="*80 + "\n")
                return "FAILED", f"Training failed (Exit code: {return_code})"
    
