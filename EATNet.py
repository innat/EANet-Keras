import tensorflow as tf 
from tensorflow.keras import layers 

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4),  embed_dim=96):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] // patch_size[0], 
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, 
                                  kernel_size=patch_size, 
                                  strides=patch_size, name='proj')
        self.norm = layers.LayerNormalization(epsilon=1e-5, name='norm')
     
    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, 
                      (H // self.patch_size[0]) * (W // self.patch_size[0]), 
                      self.embed_dim]
        )
        x = self.norm(x)
        return x
      
      
class MLP(layers.Layer):
    def __init__(self, mlp_dim, embedding_dim=None, 
                 act_layer=tf.nn.gelu, drop_rate=0.2, **kwargs):
        super(MLP, self).__init__(name='MLP', **kwargs)
        self.fc1  = layers.Dense(mlp_dim, activation=act_layer)
        self.fc2  = layers.Dense(embedding_dim)
        self.drop = layers.Dropout(drop_rate)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x
      
      
  
class ExternalAttention(layers.Layer):
    def __init__(self, dim, num_heads, dim_coefficient = 4, 
                 attention_dropout = 0,  projection_dropout = 0, 
                 **kwargs):
        super(ExternalAttention, self).__init__(name= 'ExternalAttention', **kwargs)
        self.dim       = dim 
        self.num_heads = num_heads 
        self.dim_coefficient    = dim_coefficient
        self.attention_dropout  = attention_dropout
        self.projection_dropout = projection_dropout
        
        k = 256 // dim_coefficient
        self.trans_dims = layers.Dense(dim * dim_coefficient)
        self.linear_0 = layers.Dense(k)
        self.linear_1 = layers.Dense(dim * dim_coefficient // num_heads)
        self.proj = layers.Dense(dim)
    
        self.attn_drop  = layers.Dropout(attention_dropout)
        self.proj_drop  = layers.Dropout(projection_dropout)
        
    def call(self, inputs, return_attention_scores=False, training=None):
        num_patch = tf.shape(inputs)[1]
        channel   = tf.shape(inputs)[2]
        x = self.trans_dims(inputs)
        x = tf.reshape(x, shape=(-1, 
                                 num_patch, 
                                 self.num_heads,
                                 self.dim * self.dim_coefficient // self.num_heads))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        
        # a linear layer M_k
        attn = self.linear_0(x)
        # normalize attention map
        attn = layers.Softmax(axis=2)(attn)
        # dobule-normalization
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=-1, keepdims=True))
        attn_drop = self.attn_drop(attn, training=training)
        
        # a linear layer M_v
        attn_dense = self.linear_1(attn_drop)
        x = tf.transpose(attn_dense, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [-1, num_patch, self.dim * self.dim_coefficient])
        # a linear layer to project original dim
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
  
        if return_attention_scores:
            return x, attn
        else:
            return x 
        
    def get_config(self):
        config = {
            'dim'                : self.dim,
            'num_heads'          : self.num_heads,
            'dim_coefficient'    : self.dim_coefficient,
            'attention_dropout'  : self.attention_dropout,
            'projection_dropout' : self.projection_dropout
        }
        base_config = super(ExternalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
      
      
      
class AttentionEncoder(layers.Layer):
    def __init__(self, embedding_dim, 
                 mlp_dim, num_heads, 
                 dim_coefficient,  
                 attention_dropout,  
                 projection_dropout, 
                 get_attention_matrix=False,
                 **kwargs):
        super(AttentionEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mlp_dim   = mlp_dim
        self.num_heads = num_heads
        self.dim_coefficient    = dim_coefficient
        self.attention_dropout  = attention_dropout
        self.projection_dropout = projection_dropout
        self.get_attention_matrix = get_attention_matrix
        self.mlp = MLP(mlp_dim, embedding_dim)
        
        self.etn = ExternalAttention(
            embedding_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout
        )
    
    def call(self, inputs):
        residual_1 = inputs 
        x, ext_attention_scores = self.etn(inputs, return_attention_scores=True) 
        x = layers.add([x, residual_1])
        residual_2 = x
        x = self.mlp(x)
        x = layers.add([x, residual_2])
        
        if self.get_attention_matrix:
            return x, ext_attention_scores
        else:
            return x 
    
    def get_config(self):
        config = {
            'embedding_dim'     : self.embedding_dim,
            'mlp_dim'           : self.mlp_dim,
            'num_heads'         : self.num_heads,
            'dim_coefficient'   : self.dim_coefficient,
            'attention_dropout' : self.attention_dropout,
            'projection_dropout': self.projection_dropout,
            'get_attention_matrix': self.get_attention_matrix
        }
        base_config = super(AttentionEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
