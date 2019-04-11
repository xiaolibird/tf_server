# 无菌检测 Tensorflow server
	基于 Flask、Tensorflow 和 WSGI 在 Apache 上部署的图像分析 API

## 程序结构
	程序主要分为 Apache24 库， Python 脚本。
	Apache24 库从 Apache 官网下载，为适合64位 Windows 7及以上的操作系统的Apache库，用于部署 WSGI 配置中的python Flask server。
	Python 脚本部分主要包括模型、Flask API。 模型是基于 Fine-tuned DenseNet的分类模型，输入图像，输出为分类结果（无菌、菌种1、菌种2......）。Flask API 主要包括 predict 等函数，可自行扩展编写。

## 部署步骤

- 下载(可从清华Anaconda镜像搜索) 并安装Anaconda3-4.0.0-Windows-x86_64.exe 注意选择给全部用户安装。如果系统添加环境变量失败，记得将 **%anaconda path%** 和 **%anaconda path%\Scripts** 添加到 **PATH** 环境变量中

- 安装tf和keras库，在 tf_server 文件夹下打开 powershell 输入 **pip install tensorflow==1.4.0 && pip install keras==2.1.0**

- 测试运行Flask，在 tf_server 下输入 **python tf_server.py** 看有没有在 **localhost:5000** 运行网页

- 测试运行API，在 tf_server下，在另一个 Powershell 输入 **python simple_request.py** 观察有没有返回预测结果（如果有，才继续，否则调试三个python脚本，一般是文件路径问题）

- 安装 WSGI，将 **mod_wsgi-4.5.24+ap24vc14-cp35-cp35m-win_amd64.whl** 拷贝入 **%anaconda path%\Scripts** 中，在这个文件夹打开 powershell 输入 **pip install mod_wsgi-4.5.24+ap24vc14-cp35-cp35m-win_amd64.whl**，接着输入 **mod_wsgi-express module-config**，将产生的内容 添加到 Apache24\conf\httpd.conf 

- 测试http。启动 **Apache24\bin\httpd**，访问localhost查看有没有错误，如果有，重新部署Apache

- 修改 wsgi 文件中的 **sys.path.insert(0,”%该工程绝对路径%”)**

- 配置http config。修改 **Apache24\conf\httpd.conf** 加入

   <VirtualHost *:5000>
    		ServerName example.com
    		WSGIScriptAlias / %path to wsgi file%\tf_server.wsgi
    		<Directory %path to project%>
        		Require all granted
    		</Directory>
    	</VirtualHost>

- 重启 **Apache httpd** 观察 **localhost:5000/** 和 **python simple_request.py** 的预测情况，注意各部分端口和传递参数的对应
- 部署服务。进入 **Apache24\bin** 在 Powershell **管理员权限**下运行 **httpd -k install -n tf_server** 安装服务，安装成功会有提示，以后每次开机服务会自动启动 
- 用 **python simple_request.py** 测试 如果有错误，重启服务几次一般能解决

 


## 常见问题
- 服务启动的第一次模型运算会耗费约10几秒的时间，可能会导致调用部分的timeout，长时间运行最好第一次用 python 脚本测试。
- 对于win7系统可能需要将文件路径问题改为绝对路径，且安装好服务后首次需要手动启动服务；对于查看localhost:5000时页面一直打不开的情况，需要在 **控制面板->程序和功能->打开或关闭windows功能** 勾选**Internet信息服务** 选项。


## TODO
- 从图像RGB单通道计算样品浊度。
