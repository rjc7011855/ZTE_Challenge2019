import caffe
import numpy as np
r = 240
net = caffe.Net("./fusion_model/TestModel.prototxt", "./fusion_model/TestModel.caffemodel", caffe.TEST)
weight, bias = net.params['fc5_']
U, sigma, VT = np.linalg.svd(weight.data, full_matrices=False)
#sigma[:r].sum()/sigma.sum()
net2 = caffe.Net("./fc_svd_model/TestModel.prototxt", caffe.TEST)
for key in net2.params.keys():
    if key == 'fc5_1':
        net2.params[key][0].data[...] = np.dot(np.eye(r) * sigma[:r], VT[:r])
    elif key =='fc5_':
        net2.params[key][0].data[...] = U[:,:r]
        net2.params[key][1].data[...] = bias.data
    else:
        net2.params[key][0].data[...] = net.params[key][0].data
        net2.params[key][1].data[...] = net.params[key][1].data

net2.save("./fc_svd_model/TestModel.caffemodel")

