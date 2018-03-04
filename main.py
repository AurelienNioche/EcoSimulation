import model.run
from pylab import np, plt


def main():
    r = model.run.Launcher.launch(
        x0=20,
        x1=20,
        x2=20,
        gamma=0.8,
        q=1,
        alpha=0.2,
        tau=0.01,
        t_max=500,
        single=True
    )

    plt.plot(r.indirect_exchanges)
    plt.show()

if __name__ == "__main__":
    main()