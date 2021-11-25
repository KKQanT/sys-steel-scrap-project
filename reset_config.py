from configparser import ConfigParser

if __name__ == "__main__":
    config_object = ConfigParser()
    
    config_object['taiwan_small_bigru_avgadj2'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED": "0",
        "WINDOW":"84",
        "DROPOUT":"0",
        'EPOCHS':"300",

        "N_UNITS":"[4,4]",
        "MIDDLE_DENSE_DIM":"None",

        "HEAD_SIZE":'-------',
        "NUM_HEADS":'-------',
        "FF_DIM":'-------',
        "NUM_TRANSFORMER_HEADS":'-------',
        "MLP_UNITS":'-------',
        "MLP_DROPOUT":'-------',


    }

    config_object['taiwan_gru_baseline_avg'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED":"0",
        "WINDOW":"168",
        "DROPOUT":"",
        "EPOCHS":'100',

        "N_UNITS":"2",
        "MIDDLE_DENSE_DIM":"",

        "HEAD_SIZE":'-------',
        "NUM_HEADS":'-------',
        "FF_DIM":'-------',
        "NUM_TRANSFORMER_HEADS":'-------',
        "MLP_UNITS":'-------',
        "MLP_DROPOUT":'-------',

    }

    config_object['domestic_baseline_gru_avg'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED":"0",
        "WINDOW":"168",
        "DROPOUT":"-------",
        "EPOCHS":'50',

        "N_UNITS":"2",
        "MIDDLE_DENSE_DIM":"-------",

        "HEAD_SIZE":'-------',
        "NUM_HEADS":'-------',
        "FF_DIM":'-------',
        "NUM_TRANSFORMER_HEADS":'-------',
        "MLP_UNITS":'-------',
        "MLP_DROPOUT":'-------',

    }

    config_object['domestic_bigru_avg'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED":"0",
        "WINDOW":"168",
        "DROPOUT":"0",
        "EPOCHS":'300',

        "N_UNITS":"[8,8]",
        "MIDDLE_DENSE_DIM":"None",

        "HEAD_SIZE":'-------',
        "NUM_HEADS":'-------',
        "FF_DIM":'-------',
        "NUM_TRANSFORMER_HEADS":'-------',
        "MLP_UNITS":'-------',
        "MLP_DROPOUT":'-------',

    }

    config_object['domestic_transformerv1_avgsel'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED":"0",
        "WINDOW":"168",
        "DROPOUT":"0.2",
        "EPOCHS":'500',

        "N_UNITS":"-------",
        "MIDDLE_DENSE_DIM":"-------",

        "HEAD_SIZE":'256',
        "NUM_HEADS":'4',
        "FF_DIM":'4',
        "NUM_TRANSFORMER_HEADS":'4',
        "MLP_UNITS":'[32]',
        "MLP_DROPOUT":'0.4',

    }

    with open("src/deep_learning/model_config.ini", 'w') as conf:
        config_object.write(conf)

    infer_config_object = ConfigParser()
    
    infer_config_object['taiwan_small_bigru_avgadj2'.upper()] = {
        'WINDOW':"84"
    }

    infer_config_object['taiwan_gru_baseline_avg'.upper()] = {
        'WINDOW':"168"
    }

    infer_config_object['domestic_baseline_gru_avg'.upper()] = {
        'WINDOW':"168"
    }

    infer_config_object['domestic_bigru_avg'.upper()] = {
        'WINDOW':"168"
    }

    infer_config_object['domestic_transformerv1_avgsel'.upper()] = {
        'WINDOW':"168"
    }

    with open("src/deep_learning/infer_model_config.ini", 'w') as conf:
        infer_config_object.write(conf)

    config_object = ConfigParser()

    config_object['domestic_transformerv1_avgsel_1week'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED": "0",
        "WINDOW":"168",
        "DROPOUT":"0.2",
        'EPOCHS':"500",

        "N_UNITS":"------------",
        "MIDDLE_DENSE_DIM":"-----------",

        "HEAD_SIZE":'256',
        "NUM_HEADS":'4',
        "FF_DIM":'4',
        "NUM_TRANSFORMER_HEADS":'4',
        "MLP_UNITS":'[32]',
        "MLP_DROPOUT":'0.4',


    }

    with open("src/deep_learning_1week/model_config.ini", 'w') as conf:
        config_object.write(conf)

    infer_config_object = ConfigParser()

    infer_config_object['domestic_transformerv1_avgsel_1week'.upper()] = {
        "WINDOW":"168"
    } 

    with open("src/deep_learning_1week/infer_model_config.ini", 'w') as conf:
        infer_config_object.write(conf)
    
    config_object = ConfigParser()

    config_object['domestic_transformerv1_avgsel_week1_to_4'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED": "0",
        "WINDOW":"168",
        "DROPOUT":"0.2",
        'EPOCHS':"500",

        "N_UNITS":"------------",
        "MIDDLE_DENSE_DIM":"-----------",

        "HEAD_SIZE":'256',
        "NUM_HEADS":'4',
        "FF_DIM":'4',
        "NUM_TRANSFORMER_HEADS":'4',
        "MLP_UNITS":'[32]',
        "MLP_DROPOUT":'0.4',

    }

    config_object['domestic_transformerv1_avgsel_week1_to_12'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED": "0",
        "WINDOW":"168",
        "DROPOUT":"0.2",
        'EPOCHS':"500",

        "N_UNITS":"------------",
        "MIDDLE_DENSE_DIM":"-----------",

        "HEAD_SIZE":'256',
        "NUM_HEADS":'4',
        "FF_DIM":'4',
        "NUM_TRANSFORMER_HEADS":'4',
        "MLP_UNITS":'[32]',
        "MLP_DROPOUT":'0.4',

    }

    with open("src/deep_learning_seq2seq/model_config.ini", 'w') as conf:
        config_object.write(conf)

    infer_config_object = ConfigParser()

    infer_config_object['domestic_transformerv1_avgsel_week1_to_4'.upper()] = {
        "WINDOW":"168"
    } 

    infer_config_object['domestic_transformerv1_avgsel_week1_to_12'.upper()] = {
        "WINDOW":"168"
    } 

    with open("src/deep_learning_seq2seq/infer_model_config.ini", 'w') as conf:
        infer_config_object.write(conf)