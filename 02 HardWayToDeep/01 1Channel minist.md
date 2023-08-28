# Minist基础实现

回想一下实现异或操作的神经网络，现在我们希望让网络能实现的工程更加复杂一些，比如识别一个数字。

首先，我们展示了一些minist数据。

![](../img/02/01/minist.jpg)

通过你聪明的大脑你肯定瞬间能认出这些数字，但是如果是计算机呢？如何让他们从存储时候的二进制，经过网络推理得到数字结果。
根据我们之前的工作，我们可以将图像转成一维输入，然后利用多层神经元训练学习，得到最终的10个输出，分别代表0-9的预测结果。
接下来我们就要通过这种思路实现。

首先我们可以在[这里](https://github.com/learnAI-CN/learnAI-code/blob/main/mnist.pkl.gz)下载minist数据集。
现在它是pkl.gz 格式。是一种压缩后的 Python 对象序列化文件格式。其中，pkl 表示使用 Python 的 pickle 库进行序列化，gz 表示使用 gzip 压缩算法进行压缩。


## 初始化网络
我们定义一个Network类⽤来初始化⼀个Network对象：

```python
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
        for x, y in zip(sizes[:-1], sizes[1:])]
```

通过

```python
net = Network([784, 30, 10])
```
便可以实例化一个第⼀层有 784 个神经元，第⼆层有 30 个神经元，最后层有 10 个神经元的 Network 对象。
你可以发现修改初始化时候的list就可以修改网络的层数和每层的神经元数量，这不再用对每一层进行定义。
而np.random.randn 函数本身就是用来⽣成均值为 0，标准差为 1 的⾼斯分布，可以当作参数的初始化。

## 前向传播
我们可以通过一个for循环获得前向输入。

```python
def feedforward(self, a):
    """返回神经网络的输出"""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)
    return a
```

## 随机梯度下降
现在我们要定义一个随机梯度下降算法让网络学习，

```python
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Train the neural network using mini-batch stochastic
    gradient descent. The "training_data" is a list of tuples
    "(x, y)" representing the training inputs and the desired
    outputs. The other non-optional parameters are
    self-explanatory. If "test_data" is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out. This is useful for
    tracking progress, but slows things down substantially."""
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            print "Epoch {0}: {1} / {2}".format(
            j, self.evaluate(test_data), n_test)
        else:
            print "Epoch {0} complete".format(j)
```

training_data 是⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。变量 epochs 和
mini_batch_size 正如你预料的 —— 迭代期数量，和采样时的⼩批量数据的⼤⼩。eta 是学习速率，
η。如果给出了可选参数 test_data，那么程序会在每个训练器后评估⽹络，并打印出部分进展。

## 更新参数

在每个迭代期，它⾸先随机地将训练数据打乱，然后将它分成多个适当⼤
⼩的⼩批量数据。这是⼀个简单的从训练数据的随机采样⽅法。然后对于每⼀个 mini_batch
我们应⽤⼀次梯度下降。这是通过代码 self.update_mini_batch(mini_batch, eta) 完成的，它仅
仅使⽤ mini_batch 中的训练数据，根据单次梯度下降的迭代更新⽹络的权重和偏置。

```python
def update_mini_batch(self, mini_batch, eta):
    """
    使用反向传播算法对单个小批量数据进行梯度下降，更新网络的权重和偏置。
    其中，mini_batch 是一个包含元组 (x, y) 的列表，eta 是学习率。
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
               for b, nb in zip(self.biases, nabla_b)]
```

⼤部分⼯作由这⾏代码完成：

```python
delta_nabla_b, delta_nabla_w = self.backprop(x, y)
```

这⾏调⽤了⼀个称为反向传播的算法，⼀种快速计算代价函数的梯度的⽅法。因此
update_mini_batch 的⼯作仅仅是对 mini_batch 中的每⼀个训练样本计算梯度，然后适当地更
新 self.weights 和 self.biases。

该函数的实现如下

```python
def backprop(self, x, y):
    """
    返回一个元组 (nabla_b, nabla_w)，表示代价函数 C_x 的梯度。
    nabla_b 和 nabla_w 是按层排列的 numpy 数组列表，
    类似于 self.biases 和 self.weights。
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
```

这里我们还需要定义sigmoid_prime，他是sigmoid的导数。
```python
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
```

## 数据加载

```python
def load_data():
    """
    将 MNIST 数据返回为一个元组，包含训练数据、验证数据和测试数据。
    其中，training_data 作为一个元组返回，包含两个条目。第一个条目
    包含实际的训练图像，这是一个包含 50,000 个条目的 numpy ndarray。
    每个条目又是一个包含 784 个值的 numpy ndarray，表示单个 MNIST 图像中的 28 * 28 = 784 个像素。
    training_data 元组的第二个条目是一个包含 50,000 个条目的 numpy ndarray，
    这些条目仅是元组第一个条目中对应图像的数字值（0...9）。
    validation_data 和 test_data 也类似，但每个元组仅包含 10,000 个图像。
    这是一种不错的数据格式，但为了在神经网络中使用更方便，需要对 training_data 的格式进行一些修改，
    这在下面的包装函数 load_data_wrapper() 中完成。
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    返回一个元组，包含 (training_data, validation_data, test_data)。
    该函数基于 load_data 实现，但其输出格式更适合我们实现的神经网络。
    具体而言，training_data 是一个包含 50,000 个二元组 (x, y) 的列表。
    其中，x 是一个包含输入图像的 784 维 numpy.ndarray，y 是一个表示 x 对应正确数字的 10 维 numpy.ndarray 单位向量。
    validation_data 和 test_data 分别是包含 10,000 个二元组 (x, y) 的列表。
    在每个二元组中，x 是一个包含输入图像的 784 维 numpy.ndarray，y 是与 x 对应的分类，即对应于 x 的数字值（整数）。
    显然，这意味着我们在训练数据和验证/测试数据上使用略有不同的格式。这些格式被证明是我们的神经网络代码中最方便使用的。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    返回一个 10 维的单位向量，其中第 j 个位置为 1.0，其他位置为零。
    这用于将一个数字（0...9）转换为神经网络的相应期望输出。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

```

## 开始训练

定义主函数运行代码,使⽤随机梯度下降来从 MNIST training_data 学习超过 30 次迭代期，⼩批量数
据⼤⼩为 10，学习速率 η = 3.0，

```python
if __name__=="__main__":
    # - read the input data:
    training_data, validation_data, test_data = load_data_wrapper()
    training_data = list(training_data)

    # define Network
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

经过训练可以得到以下的输出:

```python
Epoch 0 : 8277 / 10000
Epoch 1 : 8421 / 10000
...
Epoch 27 : 9468 / 10000
Epoch 28 : 9466 / 10000
Epoch 29 : 9488 / 10000
```
说明我们的结果达到了94.88%。
以上完整的代码可以在[这里](https://github.com/learnAI-CN/learnAI-code/blob/main/02-minist.py)找到




