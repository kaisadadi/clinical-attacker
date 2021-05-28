# Clinical Attacker

程序入口为attack_demo.py，其核心为attack_gate类，详细的介绍在下方展开，具体的使用示例可参看"\_\_main\_\_"中的内容。

​	(1) **init**

​	需要指定params参数，其中包含以下部分:

​	config: config文件的路径

​	model_id: 攻击的模型的id，也可以理解为选择第x个epoch的模型

​	mode: while或black，白盒或黑盒

​	gpu: 指定使用的gpu，多个话用,隔开

​	(2) **attack**

​	需要给定notes，即攻击的病历数据。notes为list，其中每个元素为单个病历，如{"text": "", "label": [0, 1, 8]}。

​	返回数据也是一个list，其中每个元素为一个字典，包含如下部分:

​	id: 对应输入数据的第几条数据(攻击失败的会自动跳过)

​	raw_seq: 原始病历文本

​	adv_seq: 对抗病历文本(为了方便观察替换处，替换的地方的左右侧均由##进行包裹)

​	raw_label: 原始标签

​	adv_label: 对抗攻击后的预测结果

​	L2_loss: 对抗攻击导致的L2损失

​	Perturbation_num: 对抗攻击产生的干扰数



config文件是程序所需的参数配置器，也是非常关键的部分。

​	(1) **栏目**

​	config文件包含train、data、model、output、attack等栏目，分别对应模型训练、数据源及预处理、模型结构、输出设置、对抗攻击设置等参数的设置。

​	举个例子，当前对抗攻击采用的是3-attack，而这个参数3位于config/default.config的attack栏目下的task中。与对抗攻击相关的参数均在config/default.config的attack栏目下。

​	(2) **参数寻找逻辑**

​	对于所有参数，均首先在指定的config路径下寻找，如示例中的"config/nlp/LSTM.config"，如果找不到会先后在"config/default_local.config"和"config/default.config"中寻找。

