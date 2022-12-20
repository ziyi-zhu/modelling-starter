def KMID(df):
    return (df.close - df.open) / df.close


def SMA(df, window):
    sma = df.close.rolling(window).mean()
    return (df.close - sma) / df.close


def pct_return(df, window):
    return df.close.pct_change(window).shift(-window)
