import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

"""
1.生成数据
2.划分训练集和测试集（验证集）
3.定义模型（线性回归模型）
4.训练模型
5.预测结果，计算误差
"""

# 1.生成数据
# linspace线性采样、均分取值
X = np.linspace(-3, 3, 300).reshape(-1, 1)  # reshape(行,列)"-1"表示N自动计算需要多少
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)  # uniform均匀分布：等概率生成数字，每个数值出现机会均等
print(X.shape, y.shape)
# 画出散点图
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c="y")
ax[1].scatter(X, y, c="y")
ax[2].scatter(X, y, c="y")
# plt.show()
# 2.划分训练集和测试集（验证集）
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.定义模型（线性回归模型）
# linear线性regression回归模型:y=wx+b
model = LinearRegression()
# 一、欠拟合
x_train1 = trainX
x_test1 = testX
# 4.训练模型
model.fit(x_train1, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型
# 打印模型参数
print(model.coef_)  # 斜率w
print(model.intercept_)  # 截距b
# 5.预测结果，计算误差
y_pred = model.predict(x_test1)  # 仅做预测，不更新模型参数
test_loss1=mean_squared_error(testY, y_pred)
train_loss1=mean_squared_error(trainY, model.predict(x_train1))
# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(X,model.predict(X),'r')
ax[0].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[0].text(-3,1.3,f"训练误差:{train_loss1:.4f}")
plt.show()