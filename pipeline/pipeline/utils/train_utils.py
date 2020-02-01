

def show_logs(cfg, epoch, results_train, results_valid):
        if cfg["params"]["verbose"] == -1 or epoch + 1 % cfg["params"]["verbose"] != 0:
            return
        header = "| train / valid | epoch "
        train = "| train | {} ".format(epoch + 1)
        valid = "| valid | {} ".format(epoch + 1)
        for key in results_train.keys():
            header += "| {} ".format(key)
            train += "| {} ".format(results_train[key])
            valid += "| {} ".format(results_valid[key])
        header += "|"
        train += "|"
        valid += "|"
        print(header)
        print(train)
        print(valid)
        print("--------------------")
