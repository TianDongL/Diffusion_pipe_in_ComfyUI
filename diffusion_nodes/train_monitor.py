import os
import sys
import subprocess
import threading
import time
import queue
import signal
import psutil
from pathlib import Path

# 全局进程管理器，确保跨实例的进程状态共享
class TensorBoardProcessManager:
    _instance = None
    _processes = {}  # port -> process_info
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_process(self, port, process, logdir, host):
        """注册TensorBoard进程"""
        self._processes[port] = {
            'process': process,
            'logdir': logdir,
            'host': host,
            'start_time': time.time()
        }
    
    def get_process(self, port):
        """获取指定端口的进程信息"""
        return self._processes.get(port)
    
    def remove_process(self, port):
        """移除进程记录"""
        if port in self._processes:
            del self._processes[port]
    
    def kill_process_on_port(self, port):
        """强制终止指定端口的TensorBoard进程"""
        try:
            # 1. 从注册表中查找
            if port in self._processes:
                process = self._processes[port]['process']
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                self.remove_process(port)
            
            # 2. 使用psutil查找并终止所有占用该端口的TensorBoard进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'tensorboard' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline'] or []
                        if any(f'--port={port}' in arg or f'--port {port}' in ' '.join(cmdline) for arg in cmdline):
                            print(f"找到占用端口{port}的TensorBoard进程: PID {proc.info['pid']}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()
                                print(f"强制终止进程 PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            return True
        except Exception as e:
            print(f"终止端口{port}上的进程时出错: {e}")
            return False
    
    def is_port_in_use(self, port):
        """检查端口是否被占用"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'tensorboard' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline'] or []
                        if any(f'--port={port}' in arg or f'--port {port}' in ' '.join(cmdline) for arg in cmdline):
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            return False
        except Exception:
            return False

class TensorBoardMonitor:
    def __init__(self):
        self.process_manager = TensorBoardProcessManager()
        self.tensorboard_process = None
        self.is_running = False
        self.log_queue = queue.Queue()
        # 添加用于存储监控信息的变量
        self.current_logdir = None
        self.current_host = None
        self.current_port = None
        self.start_time = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {
                    "forceInput": True,
                    "tooltip": "训练输出目录（来自通用训练设置）"
                }),
                "port": ("INT", {
                    "default": 6006,
                    "min": 1024,
                    "max": 65535,
                    "tooltip": "TensorBoard服务端口"
                }),
                "host": ("STRING", {
                    "default": "localhost",
                    "tooltip": "TensorBoard服务主机地址"
                }),
                "is_new_training": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否开启新训练（开启时会延迟30秒等待训练文件生成）"
                })
            },
            "optional": {
                "action": (["start", "stop", "status", "kill_port"], {
                    "default": "start",
                    "tooltip": "操作类型：启动/停止/查看状态/强制清理端口"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "status")
    FUNCTION = "execute"
    CATEGORY = "Diffusion-Pipe/Monitor"
    
    def execute(self, output_dir, port=6006, host="localhost", is_new_training=True, action="start"):
        """执行TensorBoard监控操作"""
        print("\n" + "="*80)
        print(f"[TensorBoard Monitor] 执行操作: {action}")
        print("="*80)
        
        if action == "start":
            url_result = self.start_tensorboard(output_dir, port, host, is_new_training)
            status = self.get_current_status()
            url = url_result[0] if url_result and len(url_result) > 0 else ""
            print("="*80 + "\n")
            return (url, status)
        elif action == "stop":
            result_tuple = self.stop_tensorboard()
            status = self.get_current_status()
            result = result_tuple[0] if result_tuple and len(result_tuple) > 0 else ""
            print("="*80 + "\n")
            return (result, status)
        elif action == "status":
            status = self.get_current_status()
            url = f"http://{host}:{port}" if self.is_running else ""
            print("="*80 + "\n")
            return (url, status)
        elif action == "kill_port":
            if self.process_manager.kill_process_on_port(port):
                result = f"成功清理端口{port}上的所有进程 (Successfully cleaned all processes on port {port})"
                print(result)
            else:
                result = f"清理端口{port}失败或该端口无进程 (Failed to clean port {port} or no processes on this port)"
                print(result)
            status = self.get_current_status()
            print("="*80 + "\n")
            return ("", result)
        else:
            print("="*80 + "\n")
            return ("", "未知操作 (Unknown operation)")
    
    def normalize_path(self, path):
        """规范化路径"""
        if path is None:
            return None
        if not path or path.strip() == "":
            return path
            
        # 处理Windows格式路径（在WSL环境中）
        if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
            drive_letter = path[0].lower()
            rest_path = path[3:].replace('\\', '/')
            
            if drive_letter == 'z':
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        
        # 处理反斜杠路径
        path = path.replace('\\', '/')
        
        # 确保是绝对路径
        if not path.startswith('/'):
            # 相对路径转换为绝对路径
            current_dir = Path(__file__).parent.parent
            path = str(current_dir / path)
        
        return path
    
    def find_latest_training_dir(self, base_dir):
        """在基础目录中寻找最新的训练子目录"""
        try:
            print(f"正在扫描目录: {base_dir}")
            # 寻找所有可能的训练目录
            training_dirs = []
            
            if not os.path.exists(base_dir):
                print(f"基础目录不存在: {base_dir}")
                return None
                
            items = os.listdir(base_dir)
            print(f"找到 {len(items)} 个项目")
            
            for item in items:
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    print(f"检查子目录: {item}")
                    # 检查是否包含TensorBoard日志文件
                    if self.has_tensorboard_files(item_path):
                        print(f"发现训练日志目录: {item}")
                        training_dirs.append((item_path, os.path.getmtime(item_path)))
            
            if training_dirs:
                # 按修改时间排序，返回最新的
                training_dirs.sort(key=lambda x: x[1], reverse=True)
                latest_dir = training_dirs[0][0]
                print(f"选择最新的训练目录: {latest_dir}")
                return latest_dir
            
            print("未找到包含TensorBoard日志的子目录")
            return None
            
        except Exception as e:
            print(f"查找训练目录时出错: {e}")
            return None
    
    def has_tensorboard_files(self, directory):
        """检查目录是否包含TensorBoard日志文件"""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.startswith('events.out.tfevents'):
                        return True
            return False
        except:
            return False
    
    def start_tensorboard(self, output_dir, port, host, is_new_training=True):
        """启动TensorBoard服务"""
        try:
            # 检查输入参数
            if output_dir is None:
                print("错误: output_dir 参数为 None")
                return ("",)
            
            # 检查端口是否被占用，如果是则先清理
            if self.process_manager.is_port_in_use(port):
                print(f"检测到端口{port}被占用，正在清理...")
                if self.process_manager.kill_process_on_port(port):
                    print(f"成功清理端口{port}")
                    time.sleep(2)  # 等待端口释放
                else:
                    print(f"清理端口{port}失败")
                    return ("",)
            
            # 检查是否已经在运行
            if self.is_running and self.tensorboard_process and self.tensorboard_process.poll() is None:
                url = f"http://{host}:{port}"
                return (url,)
            
            # 规范化输出目录路径
            output_dir = self.normalize_path(output_dir)
            
            # 检查并处理输出目录
            if not os.path.exists(output_dir):
                print(f"输出目录不存在: {output_dir}，正在自动创建")
                return ("",)
            
            # 如果是新训练，无条件等待30秒等待新的训练文件生成
            if is_new_training:
                print("开始新训练模式：等待新训练文件生成（30秒延迟）...")
                time.sleep(30)
                print("等待完成，开始查找最新训练目录")
            
            # 寻找最新的训练子目录
            logdir = self.find_latest_training_dir(output_dir)
            
            # 如果是非新训练模式且没找到训练日志，给出提示
            if not logdir and not is_new_training:
                print("未找到训练日志文件，跳过延迟等待")
            
            if logdir and os.path.exists(logdir):
                print(f"使用训练日志目录: {logdir}")
                final_logdir = logdir
            else:
                print(f"在目录 {output_dir} 中未找到训练日志，使用基础目录")
                final_logdir = output_dir
            
            # 构建TensorBoard命令
            cmd = [
                "tensorboard",
                f"--logdir={final_logdir}",
                f"--port={port}",
                f"--host={host}",
                "--reload_interval=30",  # 每30秒重新加载一次
                "--load_fast=false"  # 禁用实验性快速加载以提高兼容性
            ]
            
            print(f"启动TensorBoard命令: {' '.join(cmd)}")
            print(f"日志目录: {final_logdir}")
            print(f"访问地址: http://{host}:{port}")
            
            # 启动TensorBoard进程
            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True
            )
            
            # 注册到全局进程管理器
            self.process_manager.register_process(port, self.tensorboard_process, final_logdir, host)
            
            # 启动日志读取线程
            log_thread = threading.Thread(
                target=self.log_reader,
                args=(self.tensorboard_process, self.log_queue),
                daemon=True
            )
            log_thread.start()
            
            self.is_running = True
            
            # 存储当前监控信息
            self.current_logdir = final_logdir
            self.current_host = host
            self.current_port = port
            self.start_time = time.time()
            
            # 等待一小段时间检查进程是否正常启动
            time.sleep(3)
            
            if self.tensorboard_process.poll() is not None:
                # 进程已经结束，可能是启动失败
                return_code = self.tensorboard_process.returncode
                print(f"TensorBoard启动失败，返回码: {return_code}")
                
                # 尝试读取错误信息
                try:
                    stderr_output = self.tensorboard_process.stderr.read()
                    if stderr_output:
                        print(f"错误信息: {stderr_output}")
                except:
                    pass
                
                self.is_running = False
                self.process_manager.remove_process(port)
                return ("",)
            
            url = f"http://{host}:{port}"
            print(f"TensorBoard成功启动! PID: {self.tensorboard_process.pid}")
            print(f"访问地址: {url}")
            
            return (url,)
            
        except FileNotFoundError:
            print("TensorBoard未安装，请运行: pip install tensorboard")
            return ("",)
        except Exception as e:
            self.is_running = False
            print(f"启动TensorBoard时发生错误: {str(e)}")
            return ("",)
    
    def stop_tensorboard(self):
        """停止TensorBoard服务"""
        result = "TensorBoard未运行"
        
        # 优先使用全局进程管理器停止
        if self.current_port:
            if self.process_manager.kill_process_on_port(self.current_port):
                result = "TensorBoard已停止"
                print("TensorBoard已停止")
            else:
                result = "停止失败"
                print("停止TensorBoard时出现错误")
        
        # 如果当前实例还有进程引用，也尝试停止
        if self.tensorboard_process:
            try:
                # 优雅地终止进程
                self.tensorboard_process.terminate()
                
                # 等待最多5秒让进程自然退出
                try:
                    self.tensorboard_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果超时，强制杀死进程
                    self.tensorboard_process.kill()
                    self.tensorboard_process.wait()
                    print("TensorBoard进程被强制终止")
                
                if result == "TensorBoard未运行":
                    result = "TensorBoard已停止"
                    print("TensorBoard已停止")
                
            except Exception as e:
                print(f"停止TensorBoard时出现错误: {str(e)}")
                if result == "TensorBoard未运行":
                    result = f"停止失败: {str(e)}"
        
        # 清理状态
        self.tensorboard_process = None
        self.is_running = False
        self.current_logdir = None
        self.current_host = None
        self.current_port = None
        self.start_time = None
        
        return (result,)
    
    def get_current_status(self):
        """获取TensorBoard当前状态"""
        if not self.tensorboard_process:
            return "🔴 未启动"
        
        if self.tensorboard_process.poll() is None:
            # 进程仍在运行
            pid = self.tensorboard_process.pid
            
            # 计算运行时间
            if self.start_time:
                run_time = time.time() - self.start_time
                hours = int(run_time // 3600)
                minutes = int((run_time % 3600) // 60)
                seconds = int(run_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = "未知"
            
            # 构建状态信息
            status_lines = [
                f"🟢 运行中 (PID: {pid})",
                f"⏱️  运行时间: {time_str}"
            ]
            
            # 添加监控目录信息
            if self.current_logdir:
                # 显示相对路径以便阅读
                try:
                    rel_path = os.path.relpath(self.current_logdir)
                    if len(rel_path) > 80:  # 如果路径太长，显示末尾部分
                        rel_path = "..." + rel_path[-77:]
                    status_lines.append(f"📁 监控目录: {rel_path}")
                except:
                    status_lines.append(f"📁 监控目录: {self.current_logdir}")
            
            # 添加访问地址信息
            if self.current_host and self.current_port:
                status_lines.append(f"🌐 访问地址: http://{self.current_host}:{self.current_port}")
            
            # 检查监控目录中的文件
            if self.current_logdir and os.path.exists(self.current_logdir):
                try:
                    # 查找事件文件
                    event_files = []
                    for root, dirs, files in os.walk(self.current_logdir):
                        for file in files:
                            if file.startswith('events.out.tfevents'):
                                rel_file_path = os.path.relpath(os.path.join(root, file), self.current_logdir)
                                event_files.append(rel_file_path)
                    
                    if event_files:
                        status_lines.append(f"📊 发现 {len(event_files)} 个事件文件:")
                        # 只显示前3个文件，避免输出过长
                        for i, file in enumerate(event_files[:3]):
                            if len(file) > 60:  # 截断过长的文件名
                                file = file[:57] + "..."
                            status_lines.append(f"   • {file}")
                        if len(event_files) > 3:
                            status_lines.append(f"   • ... 还有 {len(event_files) - 3} 个文件")
                    else:
                        status_lines.append("⚠️  未找到TensorBoard事件文件")
                        
                except Exception as e:
                    status_lines.append(f"⚠️  读取目录时出错: {str(e)}")
            
            return "\n".join(status_lines)
        else:
            # 进程已结束
            return_code = self.tensorboard_process.returncode
            self.is_running = False
            
            if return_code == 0:
                return "🔴 已停止 (正常退出)"
            else:
                return f"🔴 已停止 (异常退出，返回码: {return_code})"
    
    def log_reader(self, process, log_queue):
        """读取TensorBoard进程的输出日志"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    log_queue.put(line)
                    print(f"[TensorBoard] {line}")
            
            for line in iter(process.stderr.readline, ''):
                if line:
                    line = line.strip()
                    log_queue.put(f"ERROR: {line}")
                    print(f"[TensorBoard Error] {line}")
                    
        except Exception as e:
            log_queue.put(f"Log reader error: {str(e)}") 