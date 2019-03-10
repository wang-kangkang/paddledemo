import paddle
import paddle.fluid as fluid
import numpy as np
def get_data():
    for i in range(20):
        print(i)
        yield [np.array([0.55]),np.array([0.66])]

image = fluid.layers.data(name='image',shape=[-1,1],dtype='float32')
label = fluid.layers.data(name='label',shape=[-1,1],dtype='float32')
reader = fluid.layers.create_py_reader_by_data(capacity=64,feed_list=[image,label])
reader.decorate_paddle_reader(paddle.batch(get_data,4))
img,label = fluid.layers.read_file(reader)
loss = (img - label) * (img - label)
adam = fluid.optimizer.Adam(0.001)
adam.minimize(loss)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
reader.start()
for i in range(3):
    fetched_img,fetched_label,fetched_loss = exe.run(fetch_list=[img,label,loss.name])
    print(fetched_img)
    print(fetched_label)
