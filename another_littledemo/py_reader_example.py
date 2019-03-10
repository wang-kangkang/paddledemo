#coding=utf-8
#http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/prepare_data/use_py_reader.html

import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist
import os
import numpy as np

def get_train_data():
    all_img=np.zeros((32,784))
    for i in range(32):
        img=np.random.random((784))
        label=np.array([np.random.randint(0,1)])
        label=label.astype(np.int64)
        all_img[i,:]=img
        if(i==31):
            print("in train", np.sum(all_img[0:8,:]))
            print("in train", np.sum(all_img[8:16,:]))
            print("in train", np.sum(all_img[16:24,:]))
            print("in train", np.sum(all_img[24:32,:]))
        yield [(img),(label)]

def get_test_data():
    all_img=np.zeros((32,784))
    for i in range(32):
        img=np.random.random((784))
        label=np.array([np.random.randint(0,1)])
        label=label.astype(np.int64)
        all_img[i,:]=img
        if(i==31):
            print("in test", np.sum(all_img[0:8,:]))
            print("in test", np.sum(all_img[8:16,:]))
            print("in test", np.sum(all_img[16:24,:]))
            print("in test", np.sum(all_img[24:32,:]))
        yield [(img),(label)]


def network(is_train):
    reader = fluid.layers.py_reader(capacity=1,shapes=((-1,784),(-1,1)),dtypes=('float32','int64'),name='train_reader' if is_train else 'test_reader',use_double_buffer=True)
    img, label = fluid.layers.read_file(reader)

    out1 = fluid.layers.fc(input = img, size=128, act='sigmoid')
    predict = fluid.layers.fc(input = out1, size=2, act = 'softmax')

    loss = fluid.layers.cross_entropy(input=predict,label=label)
    average_loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=predict,label=label)
    return average_loss, reader, predict, accuracy, img, label, loss

train_prog = fluid.Program()
train_startup = fluid.Program()
with fluid.program_guard(train_prog, train_startup):
    with fluid.unique_name.guard():
        train_loss, train_reader, train_predict, _, train_img, train_label, medium_loss = network(True)
        adam = fluid.optimizer.Adam(learning_rate=0.00001)
        adam.minimize(train_loss)

test_prog = fluid.Program()
test_startup = fluid.Program()
with fluid.program_guard(test_prog, test_startup):
    with fluid.unique_name.guard():
        test_loss, test_reader, test_predict, test_accuracy, test_img, test_label, _ = network(False)

#place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(train_startup)
exe.run(test_startup)

train_prog_compiled = fluid.CompiledProgram(train_prog)#.with_data_parallel(loss_name=train_loss.name)
test_prog_compiled = fluid.CompiledProgram(test_prog)#.with_data_parallel(share_vars_from=train_prog)

#train_reader.decorate_paddle_reader(paddle.reader.shuffle(paddle.batch(mnist.train(),64),buf_size=256))
train_reader.decorate_paddle_reader(paddle.reader.shuffle(paddle.batch(get_train_data,8),buf_size=256))
#test_reader.decorate_paddle_reader(paddle.batch(mnist.test(),64))
test_reader.decorate_paddle_reader(paddle.batch(get_test_data,8))

for epoch_id in range(10):
    train_reader.start()
    try:
        while(True):
            out_loss,out_img,out_label,out_predict, out_medium_loss = exe.run(program=train_prog_compiled, fetch_list=[train_loss,train_img,train_label,train_predict,medium_loss])
            print("train_loss",out_loss)
            print("imgsum",np.sum(out_img))
    except fluid.core.EOFException:
        print("end of train",epoch_id)
        print("\n")
        train_reader.reset()

    model_param_path="./model_parameter"
    train_param_path="./train_parameter"
    fluid.io.save_params(executor=exe,dirname=model_param_path,main_program=train_prog)
    fluid.io.save_persistables(executor=exe,dirname=train_param_path,main_program=train_prog)
        

    test_reader.start()
    try:
        while(True):
            test_out_loss,test_out_img, test_out_predict, test_out_accuracy, test_out_label = exe.run(program=test_prog_compiled, fetch_list = [test_loss,test_img,test_predict,test_accuracy, test_label])
            print("test_loss",test_out_loss)
            print("imgsum",np.sum(test_out_img))
    except fluid.core.EOFException:
        print("end of testing \n")
        test_reader.reset()

