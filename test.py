import tensorflow as tf

num_heads = 2
depth = 2

dc = tf.keras.layers.Dense(depth)
mha = tf.keras.layers.MultiHeadAttention(num_heads,depth)

target = tf.ones([8,8,depth],dtype=tf.float32)
source = tf.ones([8,8,depth],dtype=tf.float32)

y = dc(target)
print(y.shape)

print(mha(target,source).shape)

