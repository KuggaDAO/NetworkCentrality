# 文件说明

data文件为代币输入数据，具体见P大网盘，其它文件夹为对应文件名的文件运行结果

介绍顺序为发展顺序。

## 关于时间尺度

**centrality_w.py:** 对于**可重叠观察时间、三度影响\Kazt中心性完整公式**的初尝试。
输入data中文件的token文件名、观察时间T、观察起始时间s，输出在时间轴下关于两个公式cputime比较、比例最大最小值比较的图片，结果图片文件名格式为token_T_s.png，标注nw的为更改代码获得的无权图对应结果。

**centop_w.py:** 在可重叠观察时间、三度影响\Kazt中心性完整公式的基础上，加入对于**topn中心性均值比去当前图全部节点中心性均值**这一比值的考量。
输入data中文件的token文件名、观察时间T、观察起始时间s、考虑top百分比的列表percent，输出在时间轴下比值趋势的图片，结果图片文件名格式为token_T_Top.png。

**centop_nw.py:** 在可重叠观察时间、三度影响\Kazt中心性完整公式的基础上，加入对于**转化为无权图**的考量，即**仅考虑权值大于最大权值*beta的边**，实际上也减少了计算量，继续考察比值topn中心性均值/当前图全部节点中心性均值。
输入data中文件的token文件名、观察时间T、观察起始时间s、考虑top百分比的列表percent、关于权值的边选择参数beta，输出在时间轴下比值趋势的图片，结果图片文件名格式为token_b/token_T_Top_percent.png。

## 关于个体角度

**centime_w.py:** 在三度影响\Kazt中心性完整公式的基础上，加入对**时间序列**的考量，即考虑边的时间顺序，如较早时间AB互动，较晚时间BC互动，那么信息可以由A传递至C，反之不一定。
输入data中文件的token文件名、观察时间T(由起始时间开始构成互不重叠的时间区间),输出为在时间轴下关于两个公式cputime比较的曲线图、在节点编号轴下中心性的散点图，结果图片文件名格式为token/token_T.png。

**centime_nw.py:** 上述上述文件**转化为无权图的版本**。
输入为data中文件的token文件名、观察时间T(由起始时间开始构成互不重叠的时间区间)、关于权值的边选择参数beta，输出在时间轴下关于两个公式cputime比较的曲线图、在节点编号轴下中心性的散点图，结果图片文件名格式为token/token_T_b.png。

**centop_w_br.py:** 在三度影响\Kazt中心性完整公式的基础上，考察**Katz中心性的对应版本**(原考虑的broadcast为逆矩阵的行和，再考虑receive为逆矩阵的列和)，考量二者构成的二维坐标系，在中心节点的广播和接收能力都很强*(如何量化？)*的假设下可以**比较中心数量**。当然可以加入可重叠观察时间或者时间序列的考量。
输入data中文件的token文件名，输出二维坐标系的图片，结果图片文件名格式为token_n.png(其中n是当前网络规模)。

**centop_nw_br.py:** 上述文件**转化为无权图的版本**。
输入data中文件的token文件名、关于权值的边选择参数beta，输出二维坐标系的图片，结果图片文件名格式为token_b_n.png(其中n是当前网络规模)。

## package
**main.py:** 为主要脚本提供主要函数

## 主要脚本
**cenmain.py:** 利用main.py，考虑可重叠观察时间+无权图+时间序列，考虑的是Katz Receive & Katz Broadcast的top均值与全体均值的百分比变化
**individual.py:** 考虑头部个体的活跃时间段
