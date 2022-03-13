import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
print(tf.__version__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = tf.constant(1)
print(a)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = tf.constant(1, shape=(1,1))
print(a)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = tf.constant([1, 2, 3, 4])
print(b)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
с = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]], dtype=tf.float16)
print(с)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a2 = tf.cast(a, dtype=tf.float32)
print(a2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7])
v3 = tf.Variable(b)
print(v1, v2, v3, sep="\n\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v1.assign(0)
v2.assign([0, 1, 6, 7])
v3.assign_add([1, 1, 1, 1])
v1.assign_sub(5)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
val_0 = v3[0]
print(val_0)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
val_12 = v3[1:3]
print(val_12)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = tf.reshape(a, [5, 5])
b_T = tf.transpose(b, perm=[1, 0])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v22 = v2.numpy()
print(v22)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = tf.zeros((3,3))
b = tf.ones((5,3))
c = tf.ones_like(a)
d = tf.eye(3)
f = tf.fill((2, 3), -1)
print(a, b, c, d,f, sep="\n\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = tf.range(1, 11, 0.2)
print(g)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = tf.constant([1, 2, 3])
b = tf.constant([9, 8, 7])
print(a, b, sep="\n", end="\n\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1 = tf.add(a, b)       # сложение
z2 = a + b              # сложение
z1 = tf.subtract(a, b)  # вычитание
z2 = a - b              # вычитание
z1 = tf.divide(a, b)    # деление (поэлементное)
z2 = a / b              # деление (поэлементное)
z1 = tf.multiply(a, b)  # умножение (поэлементное)
z2 = a * b              # умножение (поэлементное)
z1 = a ** 2             # возведение в степень (поэлементное)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z1 = tf.tensordot(a, b, axes=0) # векторное внешнее умножение
z2 = tf.tensordot(a, b, axes=1) # векторное внутреннее умножение
print(z1, z2, sep="\n\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a2 = tf.constant(tf.range(1, 10), shape=(3, 3))
b2 = tf.constant(tf.range(5, 14), shape=(3, 3))
print(a2, b2, sep="\n\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
z1 = tf.matmul(a2, b2)  # матричное умножение
z2 = a2 @ b2            # матричное умножение
print(z1, z2, sep="\n\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = tf.tensordot(a, b, axes=0)
z = tf.reduce_sum(m) 
print(m)
print(z)
z = tf.reduce_mean(m)	     # среднее арифметическое
z = tf.reduce_max(m, axis=0)      # максимальное по столбцам
z = tf.reduce_min(m, axis=1)      # минимальное по строкам
z = tf.reduce_prod(m)             # произведение значений элементов матрицы
z = tf.square(a)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = tf.random.normal((2, 4), 0, 0.1)  # тензор 2x4 с нормальными СВ
b = tf.random.uniform((2, 2), -1, 1)  # тензор 2x2 с равномерными СВ в диапазоне [-1; 1]
c = tf.random.shuffle(range(10))  # перемешивание последовательности чисел
tf.random.set_seed(1)  # установка зерна датчика случайных чисел
d = tf.random.truncated_normal((1, 5), -1, 0.1)  # тензор 1x5 с ограниченными нормальными СВ

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tf.random.set_seed(1)
a = tf.random.normal((2, 4), 0, 0.1)
print(a)
tf.random.set_seed(1)
b = tf.random.normal((2, 4), 0, 0.1)
print(b)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2
 
df = tape.gradient(y, x)
print(df)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
x = tf.Variable([[-2.0, 1.0, 3.0]])
 
with tf.GradientTape() as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)
 
df = tape.gradient(loss, [w, b])
print(df[0], df[1], sep="\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = tf.Variable(0, dtype=tf.float32)
b = tf.constant(1.5)
 
with tf.GradientTape() as tape:
    f = (x + b) ** 2 + 2 * b
 
df = tape.gradient(f, [x, b])
print(df[0], df[1], sep="\n")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TOTAL_POINTS = 1000
 
x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)
 
k_true = 0.7
b_true = 2.0
 
y = x * k_true + b_true + noise
 
plt.scatter(x, y, s=2)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = tf.Variable(0.0)
b = tf.Variable(0.0)
EPOCHS = 500
learning_rate = 0.02
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n in range(EPOCHS):
    with tf.GradientTape() as t:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y - f))
 
    dk, db = t.gradient(loss, [k, b])
 
    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)
print(k, b, sep="\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE
 
for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
 
        with tf.GradientTape() as t:
            f = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - f))
 
        dk, db = t.gradient(loss, [k, b])
 
        k.assign_sub(learning_rate * dk)
        b.assign_sub(learning_rate * db)
print(k, b, sep="\n")        
  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE
opt = tf.optimizers.SGD(learning_rate=0.02) 
#opt = tf.optimizers.SGD(momentum=0.5, learning_rate=0.02)  # Метод моментов
#opt = tf.optimizers.SGD(momentum=0.5, nesterov=True, learning_rate=0.02) # Метод Нестерова
#opt = tf.optimizers.Adagrad(learning_rate=0.1) # Adagrad
#opt = tf.optimizers.Adadelta(learning_rate=1.0)  # Adadelta
#opt = tf.optimizers.RMSprop(learning_rate=0.01)
#opt = tf.optimizers.Adam(learning_rate=0.1)
 

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
 
        with tf.GradientTape() as t:
            f = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - f))
 
        dk, db = t.gradient(loss, [k, b])
 
        #k.assign_sub(learning_rate * dk)
        #b.assign_sub(learning_rate * db)
        opt.apply_gradients(zip([dk, db], [k, b]))
print(k, b, sep="\n")        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%  Реализация полносвязного слоя нейронов
# outputs – число выходов слоя (число нейронов)
# fl_init – флаг начальной инициализации весовых коэффициентов
# __call__ превратит наш класс DenseNN в функтор
# (то есть, мы его сможем использовать подобно функции)
#
# x - вектор входного сигнала
#

class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False
 
    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
 
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
 
            self.fl_init = True
 
        y = x @ self.w + self.b
        return y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Создаём модель для обучения алгебраическому сложению
# Количество входов определяет сама по размерности входного вектора
# Столько весов w и создаёт
model = DenseNN(1)
print(model(tf.constant([[1.0, 2.0]])) )
print(model.w)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%% Готовим обучающую выборку
x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))
y_train = [a + b for a, b in x_train]
#%%%%% Затем, зададим функцию потерь и оптимизатор для градиентного спуска:
loss = lambda x, y: tf.reduce_mean(tf.square(x - y))
opt = tf.optimizers.Adam(learning_rate=0.01)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EPOCHS = 50
for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)
        y = tf.constant(y, shape=(1, 1))
 
        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))
 
        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
 
        print(f_loss.numpy())
        
# Этот цикл обучения можно заменить методом fit        
# model.fit(x_train, y_train, epochs=50)        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print(model.trainable_variables)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
