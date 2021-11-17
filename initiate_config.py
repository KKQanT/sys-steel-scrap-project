from configparser import ConfigParser

if __name__ == "__main__":
    config_object = ConfigParser()
    
    config_object['taiwan_small_bigru_avgadj2'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED": "0",
        "WINDOW":"84",
        "N_UNITS":"[4,4]",
        "MIDDLE_DENSE_DIM":"None",
        "DROPOUT":"0"
    }

    config_object['taiwan_gru_baseline_avg'.upper()] = {
        "SPLIT_PCT":"20",
        "SEED":0,
        "WINDOW":"168",
        "N_UNITS":2,
        "MIDDLE_DENSE_DIM":"",
        "DROPOUT":""
    }

    with open("src/deep_learning/model_config.ini", 'w') as conf:
            config_object.write(conf)