import matplotlib.pyplot as plt

def plot_forecast(history,forecast,title="Energy Load Forecas"):

    plt.figure(figsize=(14,6))
    plt.plot(history,label="History")
    plt.plot(forecast,label="Forecast",color="orange")
    plt.title(title)
    plt.legend()
    plt.show()

def save_results_to_csv(forecast,path="forecast_output.csv"):
    forecast.to_csv(path)    