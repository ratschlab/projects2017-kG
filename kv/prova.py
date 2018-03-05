import tensorflow as tf

c = []
g = []
kk = 0
for d in ['/device:GPU:2', '/device:GPU:3']:
    g.append(tf.Graph())
    with g[kk].as_default():
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
            d = tf.constant([1.,1.,1.,1.], shape = [2,2])
            for i in range(0,1000):
                tf.matmul(c[0],d)
            

    kk+=1

def f((k,g)):
    with g[k].as_default():
        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if k == 0:
            print 'k = 0 have'
            print(sess.run(c))
        else:
            print 'k = 1 have'
            print(sess.run(a))
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

pool = ThreadPool(5)
job_args = [(i,g) for i in range(0,2)]
pool.map(f, job_args)
pool.close()
pool.join()
