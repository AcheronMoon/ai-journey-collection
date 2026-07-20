import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.metrics import mean_squared_error  # metrics(评估工具包):MSE均方误差损失函数

"""
1.生成数据
2.划分训练集和测试集（验证集）
3.定义模型（线性回归模型）
4.训练模型
5.预测结果，计算误差
"""

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1.生成数据
# linspace线性采样、均分取值
X = np.linspace(-3, 3, 300).reshape(-1, 1)  # reshape(行,列)"-1"表示N自动计算需要多少
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)  # uniform均匀分布：等概率生成数字，每个数值出现机会均等
# print(X.shape, y.shape)  # (300,1) (300,1)
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
# print(x_train1.shape, x_test1.shape)  # (240,1) (60,1)

# 4.训练模型
model.fit(x_train1, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型
# # 打印模型参数
# print(model.coef_)  # 斜率w
# print(model.intercept_)  # 截距b

# 5.预测结果，计算误差
y_pred = model.predict(x_test1)  # 仅做预测，不更新模型参数
test_loss1 = mean_squared_error(testY, y_pred)  # MSE均方误差损失函数
train_loss1 = mean_squared_error(trainY, model.predict(x_train1))
# 画出拟合曲线，并标出训练误差和测试误差
ax[0].plot(X, model.predict(X), 'r')
ax[0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练误差:{train_loss1:.4f}")
# plt.show()

# 二、恰好拟合(5次多项式)
poly5 = PolynomialFeatures(degree=5)  # 数据预处理转换器;degree=5:5次多项式特征，共6个（包含0次项）
x_train2 = poly5.fit_transform(trainX)
x_test2 = poly5.fit_transform(testX)
# print(x_train2.shape, x_test2.shape)  # (240,6) (60,6)

# 4.训练模型
model.fit(x_train2, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型
# # 打印模型参数
# print(model.coef_)  # 斜率w
# print(model.intercept_)  # 截距b

# 5.预测结果，计算误差
y_pred2 = model.predict(x_test2)  # 仅做预测，不更新模型参数
test_loss2 = mean_squared_error(testY, y_pred2)  # MSE均方误差损失函数
train_loss2 = mean_squared_error(trainY, model.predict(x_train2))
# 画出拟合曲线，并标出训练误差和测试误差
ax[1].plot(X, model.predict(poly5.fit_transform(X)), 'r')
ax[1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练误差:{train_loss2:.4f}")
# plt.show()

# 三、过拟合(25次多项式)
poly25 = PolynomialFeatures(degree=25)  # 数据预处理转换器;degree=5:5次多项式特征，共6个（包含0次项）
x_train3 = poly25.fit_transform(trainX)
x_test3 = poly25.fit_transform(testX)
# print(x_train2.shape, x_test2.shape)  # (240,6) (60,6)

# 4.训练模型
model.fit(x_train3, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型
# # 打印模型参数
# print(model.coef_)  # 斜率w
# print(model.intercept_)  # 截距b

# 5.预测结果，计算误差
y_pred3 = model.predict(x_test3)  # 仅做预测，不更新模型参数
test_loss3 = mean_squared_error(testY, y_pred3)  # MSE均方误差损失函数
train_loss3 = mean_squared_error(trainY, model.predict(x_train3))
# 画出拟合曲线，并标出训练误差（经验误差）和测试误差（泛化误差）
ax[2].plot(X, model.predict(poly25.fit_transform(X)), 'r')
ax[2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练误差:{train_loss3:.4f}")

ax[0].set_title("欠拟合")
ax[1].set_title("恰好拟合")
ax[2].set_title("过拟合")
plt.show()