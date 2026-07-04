
适用于任何 Ubuntu / Debian / 其他 Linux 服务器，通过本地 Clash 代理实现科学上网，同时完成 .codex/auth.json 等认证文件的复制。

0️⃣ 前置准备：确认本地代理正常

先确保你本地的 Clash 能联网。在本地电脑终端执行：
```bash
curl -x http://127.0.0.1:7897 -I https://www.google.com
```
若返回 HTTP/2 200 或 301 / 308，说明代理可用 ✅
```bash
HTTP/1.1 200 Connection established
HTTP/1.1 200 OK
```
Clash 默认端口(不同的vpn软件，自行查看)：
HTTP(S)：7897
SOCKS5：7898
可在 Clash Dashboard 查看端口。


1️⃣ 建立 SSH 反向代理隧道
关键步骤：让远程服务器“反向”访问你本地的代理。在本地电脑执行(win+r cmd，下面详细在远程服务器查看)：
```bash
ssh -p <服务器端口> \
    -R 9567:127.0.0.1:9567 \
    -o ServerAliveInterval=60 \
    root@<服务器地址>
```

示例：
复制成功
```bash
C:\Users\chensheng>ssh -p 25327 -R 9567:127.0.0.1:9567 -o ServerAliveInterval=60 root@region-9.autodl.pro
```

参数含义：
- ssh
SSH 客户端命令，作用是通过 SSH 加密协议远程登录连接服务器。
- -p 25327
对应你表格里的-p参数：指定远程 SSH 服务的端口号。
常规 SSH 默认端口是 22，这里远程服务器region-9.autodl.pro把 SSH 端口改成了25327，必须通过这个端口才能连上。
- -R 9567:127.0.0.1:9567（远程端口转发，核心功能）
格式规范：-R 远程端口:本地IP:本地端口
含义：在远端服务器（也就是region-9.autodl.pro这台机器）开放 9567 端口；只要访问远端服务器的127.0.0.1:9567，流量就会经由 SSH 隧道转发回你自己本地电脑的127.0.0.1:9567。
典型用处：比如你本地 Clash 代理监听了本机 9567 端口，远程服务器就可以借助这条隧道走你本地的代理网络。
- -o ServerAliveInterval=60
-o用于单独设置 SSH 高级配置项，这里配置ServerAliveInterval=60：
SSH 客户端每隔60 秒主动向远端服务器发送心跳探测包，避免长时间没有数据交互时，中间路由器、防火墙把 SSH 空闲连接断开，起到保活防掉线的效果。
- root@region-9.autodl.pro
root：登录远程服务器使用的用户名（超级管理员账号）
region-9.autodl.pro：远程服务器的域名（也可以换成 IP 地址），这就是你要建立 SSH 连接的目标主机。



2️⃣ 在远程服务器上设置代理环境变量
2.1 临时设置（当前会话生效）
在远程终端执行：

复制成功
```bash
export http_proxy="http://127.0.0.1:9567"
export https_proxy="http://127.0.0.1:9567"
export no_proxy="localhost,127.0.0.1,.local"
```
测试：
```bash
curl -I https://chat.openai.com
curl -I https://www.google.com
```
如返回 200 / 30x 响应，即代理生效 ✅

2.2 永久生效（登录自动加载）
将上述环境变量追加到 shell 配置文件。

bash
```bash
echo 'export http_proxy="http://127.0.0.1:9567"' >> ~/.bashrc
echo 'export https_proxy="http://127.0.0.1:9567"' >> ~/.bashrc
echo 'export no_proxy="localhost,127.0.0.1,.local"' >> ~/.bashrc
source ~/.bashrc
```

zsh
```zsh
echo 'export http_proxy="http://127.0.0.1:7897"' >> ~/.zshrc
echo 'export https_proxy="http://127.0.0.1:7897"' >> ~/.zshrc
echo 'export no_proxy="localhost,127.0.0.1,.local"' >> ~/.zshrc
source ~/.zshrc
```

3️⃣ 测试代理是否生效
在远程服务器执行：
```bash
curl -I https://www.google.com
curl -I https://chat.openai.com
```
若输出包含 HTTP/1.1 200 Connection established 或 HTTP/2 200，说明代理成功。

其他命令测试示例：
```bash
pip install requests
apt update
git clone https://github.com/openai/openai-cookbook.git
```

4️⃣ VS Code Remote SSH 自动化配置

config文件
点击左下角ssh，连接到主机，配置SSH主机
```
# AutoDL
Host region-9.autodl.pro
    HostName region-9.autodl.pro
    Port 25327
    User root
    RemoteForward 9567 127.0.0.1:9567
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

5️⃣ 远程服务器codex文件配置，先下载codex插件

vscode下载codex插件才会生成.codex文件

5.1 复制 .codex/auth.json（或其他认证文件）
若远程服务器需要访问 ChatGPT / Codex 等服务，需要同步认证文件。

假设本机路径为 ~/.codex/auth.json：

拷贝auth.json到vscode的.codex文件，这是在本地.codex文件中

5.2  修改.bashrc 文件
将下面配置放在文件的最底下，记得改成自己的端口号

# ========== 代理配置 ==========
```bash
PROXY_SERVER="127.0.0.1"
PROXY_PORT="9567"
NO_PROXY_LIST="localhost,127.0.0.1,.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# Bash原生端口检测：零依赖、全系统兼容、毫秒级完成无卡顿

if (echo > /dev/tcp/$PROXY_SERVER/$PROXY_PORT) 2>/dev/null; then
    echo "代理可用，使用 HTTP 代理"
    export http_proxy="http://$PROXY_SERVER:$PROXY_PORT"
    export https_proxy="http://$PROXY_SERVER:$PROXY_PORT"
    export HTTP_PROXY="http://$PROXY_SERVER:$PROXY_PORT"
    export HTTPS_PROXY="http://$PROXY_SERVER:$PROXY_PORT"
    export no_proxy="$NO_PROXY_LIST"
    export NO_PROXY="$NO_PROXY_LIST"
else
    echo "代理不可用，切换到直连"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
fi

# export http_proxy=http://127.0.0.1:9567;  #HTTP
# export https_proxy=http://127.0.0.1:9567; #HTTPS
# export no_proxy="localhost,127.0.0.1,.local"
```

5.3  创建config.toml
写入
```bash
[projects."/root"]
trust_level = "trusted"

[proxy]
http_proxy="http://127.0.0.1:9567"
https_proxy="http://127.0.0.1:9567"
```





