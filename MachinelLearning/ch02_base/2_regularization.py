import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.metrics import mean_squared_error  # metrics(评估工具包):MSE均方误差损失函数

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.生成数据
# linspace线性采样、均分取值
X = np.linspace(-3, 3, 300).reshape(-1, 1)  # reshape(行,列)"-1"表示N自动计算需要多少
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)  # uniform均匀分布：等概率生成数字，每个数值出现机会均等
# print(X.shape, y.shape)  # (300,1) (300,1)
# 画出散点图
fig, ax = plt.subplots(2, 3, figsize=(15, 4))
ax[0,0].scatter(X, y, c="y")
ax[0,1].scatter(X, y, c="y")
ax[0,2].scatter(X, y, c="y")


# 2.划分训练集和测试集（验证集）
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 过拟合(20次多项式)
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(trainX)
x_test3 = poly20.fit_transform(testX)

#一、不加正则化
# 3.定义模型
model = LinearRegression()

# 4.训练模型
model.fit(x_train3, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型

# 5.预测结果，计算误差
y_pred1 = model.predict(x_test3)  # 仅做预测，不更新模型参数
test_loss1 = mean_squared_error(testY, y_pred1)  # MSE均方误差损失函数
train_loss1 = mean_squared_error(trainY, model.predict(x_train3))
# 画出拟合曲线，并标出训练误差（经验误差）和测试误差（泛化误差）
ax[0,0].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[0,0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0,0].set_title("线性回归")
ax[1,0].bar(np.arange(21),model.coef_.reshape(-1))

# 二、L1正则化
# 3.定义模型
model = Lasso(alpha=0.01)  #  alpha:正则化系数

# 4.训练模型
model.fit(x_train3, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型

# 5.预测结果，计算误差
y_pred2 = model.predict(x_test3)  # 仅做预测，不更新模型参数
test_loss2 = mean_squared_error(testY, y_pred2)  # MSE均方误差损失函数
train_loss2 = mean_squared_error(trainY, model.predict(x_train3))
# 画出拟合曲线，并标出训练误差（经验误差）和测试误差（泛化误差）
ax[0,1].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[0,1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[0,1].set_title("Lasso回归")
ax[1,1].bar(np.arange(21),model.coef_.reshape(-1))

# 三、L2正则化
# 3.定义模型
model = Ridge(alpha=0.2)

# 4.训练模型
model.fit(x_train3, trainY)  # .fit("训练集特征","训练集标签")只用训练数据训练模型

# 5.预测结果，计算误差
y_pred3 = model.predict(x_test3)  # 仅做预测，不更新模型参数
test_loss3 = mean_squared_error(testY, y_pred3)  # MSE均方误差损失函数
train_loss3 = mean_squared_error(trainY, model.predict(x_train3))
# 画出拟合曲线，并标出训练误差（经验误差）和测试误差（泛化误差）
ax[0,2].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[0,2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[0,2].set_title("Ridge回归")
ax[1,2].bar(np.arange(21),model.coef_.reshape(-1))
plt.show()
