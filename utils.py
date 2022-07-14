from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def loop(test_model, act, iters):
    test_losses = []
    test_accs = []
    for i in tqdm(range(iters)):
        test_loss , test_acc = test_model(act)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return test_losses , test_accs

def plot_histograms(test_losses, test_accs):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.hist(test_losses, bins=20)
    plt.title(f"Average Test Losses {round(np.average(test_losses), 4)}, \n std:{round(np.std(test_losses),4)}")
    plt.subplot(1,2,2)
    plt.hist(test_accs, bins=20)
    plt.title(f"Average Test Accuracy = {round(np.average(test_accs), 4)}, \n std = {round(np.std(test_accs), 4)}")
    plt.show()


