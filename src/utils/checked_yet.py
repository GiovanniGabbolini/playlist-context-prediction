def checked_yet(architecture, method, params):
    path = f"{preprocessed_dataset_path}/theme_prediction/mpd/results/{architecture}/{method}/results.json"
    return_value = False
    try:
        with open(path) as f:
            l = json.load(f)
        for d in l:
            found = True
            for k in params.keys():
                try:
                    found = found and d[k] == params[k]
                except KeyError:
                    found = False
            if found:
                return_value = True
                break
    except FileNotFoundError:
        pass
    return return_value
