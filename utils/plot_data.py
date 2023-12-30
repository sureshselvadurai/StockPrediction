from matplotlib import pyplot as plt


class Plot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def add_data_series(self, x_axis, y_axis, label, color, linestyle='-'):
        y_axis = [float(value) for value in y_axis]
        self.ax.plot(x_axis, y_axis, label=label, color=color, linestyle=linestyle)

    def show_plot(self, symbol):
        self.ax.set_title(f'{symbol} LSTM Model Performance')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Stock Price')
        self.ax.legend()
        self.fig.tight_layout()
        # plt.show()

    def save_plot(self, filename, file_format='png'):
        self.fig.savefig(filename, format=file_format)

