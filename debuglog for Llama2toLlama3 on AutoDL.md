# matplotlib 无微软雅黑字体问题

## Step1: 下载开源中文字体

```bash
# 更新包列表并安装文泉驿系列字体（包含微米黑和正黑）
sudo apt-get update
sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei ttf-wqy-zenhei
```

安装完成后检查系统能够识别：

```bash
fc-list :lang=zh
```

## Step2: 清理缓存

```bash
# 查看 Matplotlib 缓存路径
python -c "import matplotlib; print(matplotlib.get_cachedir())"

# 删除该目录下的所有内容
rm -rf $(python -c "import matplotlib; print(matplotlib.get_cachedir())")/*
```

## Step3: 重启内核，在环境下运行代码

检查 matplotlib 能否检测到字体。如果输出了 WenQuanYi Micro Hei 则能够正常运行。

```python
from matplotlib import font_manager
fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
for name in fonts:
    if 'Wen' in name or 'wqy' in name.lower() or 'CJK' in name:
        print(name)
```

## Step4: 在首次 import 时进行设置

```python
import matplotlib.pyplot as plt

# 设置字体，按优先级排列
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Microsoft YaHei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
```

## 如果还不行，清除缓存

```bash
rm -rf ~/.cache/matplotlib
```

然后重启内核。如果还有问题直接将下方显式指定字体路径的代码写进程序中。

```python
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 直接把字体文件注册给 Matplotlib
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False
```