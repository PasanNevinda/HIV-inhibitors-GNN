
GNN_GAT_PARAMS = {
    #model
    "model_embed_size" : 64,
    "model_num_heads" : 3,
    "model_dropout_gat" : 0.6,
    "model_num_layers" : 3,
    "model_topKratio" : 0.8,
    "model_dense_neurons" : 256,
    "model_dense_dropout_rate" : 0.5,

    #optimizer
    "sgd_learning_rate" : 0.01,
    "sgd_momentem" : 0.9,
    "sgd_weight_dacay" : 0.0001,

    #other
    "batch_size" : 64
}