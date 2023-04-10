import tensorflow as tf

#num_heads = 2
#depth = 2
#
#dc = tf.keras.layers.Dense(depth)
#mha = tf.keras.layers.MultiHeadAttention(num_heads,depth)
#
#target = tf.ones([8,8,depth],dtype=tf.float32)
#source = tf.ones([8,8,depth],dtype=tf.float32)
#
#y = dc(target)
#print(y.shape)
#
#print(mha(target,source).shape)

#bnorm = tf.keras.layers.BatchNormalization()
#lnorm = tf.keras.layers.LayerNormalization()

##########################################
#x = tf.zeros([2,3,5])
#x = tf.constant(
#    [[[1,2],
#      [3,4]],
#     [[0,0],
#      [0,0]]]
#    ,dtype=tf.float32)
#x = tf.constant(
#    [[1,2],
#     [0,0]]
#    ,dtype=tf.float32)
#x = tf.constant(
#    [[1.0, 2.0, 3.0],
#    [1.0, 2.0, 3.0],
#    [3.0, 2.0, 1.0],
#    [3.0, 2.0, 1.0]]
#)
#print('==== input ====')
#print(x)

##print('==== batch norm ====')
##print(bnorm(x,training=True))
#print('==== layer norm ====')
#print(lnorm(x,training=True))
#lnorm(x,training=True)
#lnorm(x,training=True)
#lnorm(x,training=True)
#lnorm(x,training=True)
#print(lnorm(x,training=False))

##########################################
#x = tf.zeros([2,3,5])
#x = tf.constant(
#    [[[1,0],
#      [2,0]],
#     [[3,0],
#      [4,0]]]
#    ,dtype=tf.float32)
#x = tf.constant(
#    [[1,0],
#     [3,0]]
#    ,dtype=tf.float32)
#print('==== input ====')
#print(x)

#print('==== batch norm ====')
#print(bnorm(x,training=True))
#bnorm(x,training=True)
#bnorm(x,training=True)
#bnorm(x,training=True)
#bnorm(x,training=True)
#print(bnorm(x,training=False))
#print('==== layer norm ====')
#print(lnorm(x,training=True))

##########################################
#x = tf.constant(
#    [[[1,0],
#      [0,0]],
#     [[2,0],
#      [0,0]]]
#    ,dtype=tf.float32)
#print('==== input ====')
#print(x)
#print('==== batch norm rindow ====')
#mean = tf.math.reduce_mean(x,axis=0)
#norm = (x-mean) / tf.math.sqrt(tf.math.reduce_mean(tf.math.square(x-mean),axis=0)+1e-7)
#print(norm)
##########################################
#x = tf.constant(
#    [[[1,2],
#      [0,0]],
#     [[0,0],
#      [0,0]]]
#    ,dtype=tf.float32)
##print('==== input ====')
##print(x)
#print('==== layer norm rindow ====')
#size = tf.shape(x)[1]
#mean = tf.math.reduce_mean(x,axis=-1,keepdims=True)
#print(mean)
#mean = tf.repeat(mean,[size],axis=-1)
#print(x-mean)
#print(tf.math.square(x-mean))
#var = tf.math.reduce_mean(tf.math.square(x-mean),axis=-1,keepdims=True)
#var = tf.repeat(var,[size],axis=-1)
#print(var)
#norm = (x-mean) / tf.math.sqrt(var+1e-7)
#print(norm)

#print(bnorm.trainable_variables)
#print(lnorm.trainable_variables)
#print('======bnorm weights==========')
#for w in bnorm.weights:
#  print(w)
#print('======lnorm weights==========')
#for w in lnorm.weights:
#  print(w)

################################################################

atn = tf.keras.layers.Attention()
q = tf.Variable(tf.ones([2,3,4]))
v = tf.Variable(tf.ones([2,5,4]))
k = tf.Variable(tf.ones([2,5,4]))
qmask = tf.constant([ # (2,3)
    [True,True,False],
    [True,False,False],
])
vmask = tf.constant([ # (2,5)
    [True,True,False,False,False],
    [True,True,True,True,False],
])
with tf.GradientTape() as tape:
  output,scores = atn([q,v,k],mask=[qmask,vmask],return_attention_scores=True)
grads = tape.gradient(output,[q,v,k])
print('scores=',scores)
print('output=',output)
print('dQ=',grads[0])
print('dV=',grads[1])
