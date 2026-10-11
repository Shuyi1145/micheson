# 运行程序

在香橙派 Ubuntu 桌面或远程桌面的终端中，打开本项目文件夹。

首次安装依赖：

```bash
sudo apt update
sudo apt install python3-opencv python3-tk python3-pil.imagetk python3-numpy
```

先运行演示模式（不需要摄像头）：

```bash
python3 main.py --demo
```

关闭演示窗口，插入 USB 摄像头后运行：

```bash
python3 main.py
```

摄像头打不开时，可尝试：

```bash
python3 main.py --camera 1
```

测量步骤：

1. 单击画面选择干涉条纹中心。
2. 输入千分尺起始读数（mm），点击“开始记录”。
3. 单向、缓慢移动反射镜 M2。
4. 点击“停止并分析”，输入终止读数（mm）。
5. 点击“按当前读数计算并保存”。结果保存在 `data` 文件夹。

注意：普通 SSH 不直接显示程序窗口；摄像头应关闭自动曝光和自动增益，避免过曝；运动太快会漏计。千分尺分辨率未知时留空。

