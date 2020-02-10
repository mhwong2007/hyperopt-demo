from model.eval import calc_auc
from model.optimise import optimise
from model.train import train_model
from utils.read_data import load_data


if __name__ == '__main__':
    train, test = load_data()

    # define how many round we want to try
    max_evals = 10
    best = optimise(train, max_evals)

    # show the best parameter
    print(best)

    # train our model
    bst = train_model(train, **best)

    # testing
    y = test.get_label()
    y_pred = bst.predict(test)

    # show auc
    auc = calc_auc(y, y_pred)
    print(auc)


