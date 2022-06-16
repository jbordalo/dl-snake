import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animator:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ims = []
        self.ax.axis('off')
        self.fig.set_size_inches((10, 10), forward=True)
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    def add_to_animation(self, board):
        im = self.ax.imshow(board, animated=True)
        if len(self.ims) == 0:
            self.ax.imshow(board)
        self.ims.append([im])

    def save_animation(self, name):
        plt.gca().text(3, 3, name, fontsize=30, color='yellow')
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, blit=True, repeat_delay=1000)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(f"animations/{name}.mp4", writer=writer)
        plt.close()
