from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def find_best_sarima_params(series,seasonal_period=365):

    model_auto = auto_arima(
        series,
        seasonal=True,
        m=seasonal_period,
        trace=True,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True
    )

    return model_auto.order,model_auto.seasonal_order

def train_sarima_model(series,order,seasonal_order):

    model = SARIMAX(
        series,
        order=order,
        seasonal_order = seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        
    )

    model_fit = model.fit()

    return model_fit

def forecast_future(model_fit,steps = 30):

    return model_fit.forecast(steps)