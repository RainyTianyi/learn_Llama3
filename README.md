本仓库用于在 nanoGPT 的基础上学习 Llama3 的架构。两个核心文件都在 AutoDL 上跑通，需要把 .ipynb 文件和 .py 文件传到 AutoDL 的数据盘中再运行，即路径 /root/autodl-tmp 下。使用的环境为：

```
镜像    PyTorch  2.1.2  Python  3.10(ubuntu22.04)   CUDA  11.8
GPU     vGPU-32GB(32GB) * 1
CPU     25 vCPU Intel(R) Xeon(R) Platinum 8481C
内存    90GB
```