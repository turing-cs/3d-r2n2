import tensorflow as tf

LR, Ep = 0.01,100

#Build tensorflow graph
X = tf.constant([1.0],[2.0],[3.0])
T = tf.constant([3.0],[2.0],[1.0])
W = tf.Variable(tf.random_uniform_initializer([3,3]))
Y = tf.matmul(W,X)
L = tf.reduce_mean(tf.abs(T-Y))

GD = tf.compat.v1.train.GradientDescentOptimizer(LR)
Gradients = GD.compute_gradients(L)
TRAIN_STEP = GD.apply_gradients(Gradients)

with tf.session() as sess:
    tf.global_variables_initializer().run()
    for i in range(Ep):
        results = sess.run([L,Y,TRAIN_STEP])

print("Loss : ", result[0])
print("Prediction :", results[1])
