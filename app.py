import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import tkinter as tk
from tkinter import Toplevel, Entry, filedialog
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import font
import requests
import os
import zipfile
import shutil

plt.style.use('ggplot')

GITHUB_REPO = "markhen2/M_CARLO_APP"
CURRENT_VERSION = "1.0.2"

def get_latest_release():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def check_for_updates():
    latest_release = get_latest_release()
    latest_version = latest_release["tag_name"]
    if latest_version != CURRENT_VERSION:
        download_url = latest_release["zipball_url"]
        download_and_apply_update(download_url)
    else:
        print("No updates available.")

def download_and_apply_update(url):
    response = requests.get(url)
    response.raise_for_status()

    temp_zip_path = "temp_update.zip"
    with open(temp_zip_path, "wb") as temp_zip_file:
        temp_zip_file.write(response.content)
    
    temp_dir = "temp_update_dir"
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    

    specific_files = ["app.py"] 
    for file_name in specific_files:
        src_path = os.path.join(temp_dir, file_name)
        dest_path = os.path.join("desired_location", file_name)  
        shutil.copy(src_path, dest_path)
    
    os.remove(temp_zip_path)
    shutil.rmtree(temp_dir)

    print("Update applied successfully.")
def apply_update(update_dir):
    for item in os.listdir(update_dir):
        s = os.path.join(update_dir, item)
        d = os.path.join(os.getcwd(), item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("Update applied. Please restart the application.")

def monte_carlo(ticker, prediction_days, number_simulations,BSM=0.26,threshold1=None,threshold2=None):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365) #Used to set a 1 yr period
    df = yf.download(ticker, start=start_date, end=end_date) 
    log_returns = np.log(df['Close'] / df['Close'].shift(1)) #Log returns
    days_to_forecast = prediction_days
    num_simulations = number_simulations
    dt = 1  # 1 trading day
    if threshold2 is None:
        threshold2 = df['Adj Close'][-1]*1.9
    if threshold1 is None:
        threshold1 = df['Adj Close'][-1]*0.5

    volatility_bsm =BSM  # Assuming 26% annualized volatility
    print("BSM Implied Volatility: ", volatility_bsm)
    def run_simulation(volatility, dt, annualized=False):
        simulated_prices = np.zeros((days_to_forecast, num_simulations))
        simulated_prices[0] = df['Close'][-1]
        if annualized:
            volatility = volatility / np.sqrt(252)
        for t in range(1, days_to_forecast):
            random_walk = np.random.normal(loc=log_returns.mean() * dt,
                                           scale=volatility * np.sqrt(dt),
                                           size=num_simulations)
            simulated_prices[t] = simulated_prices[t - 1] * np.exp(random_walk)
        return simulated_prices

    simulated_prices_bsm = run_simulation(volatility_bsm, dt, annualized=True)
    fig, ax = plt.subplots(figsize=(15, 7))
    mean_price_path = np.mean(simulated_prices_bsm, axis=1)
    median_price_path = np.median(simulated_prices_bsm, axis=1)
    lower_bound_68 = np.percentile(simulated_prices_bsm, 16, axis=1)
    upper_bound_68 = np.percentile(simulated_prices_bsm, 84, axis=1)
    lower_bound_95 = np.percentile(simulated_prices_bsm, 2.5, axis=1)
    upper_bound_95 = np.percentile(simulated_prices_bsm, 97.5, axis=1)
    ax.plot(simulated_prices_bsm, color='gray', alpha=0.3, linewidth=1) # This is the plot for the randomised samples
    ax.plot(mean_price_path, color='black', label='Mean Price Path', linewidth=2) #MEan path plot
    ax.plot(median_price_path, color='blue', label='Median Price Path', linewidth=2)#Median
    ax.fill_between(range(days_to_forecast), lower_bound_68, upper_bound_68, color='green', alpha=0.3, label='68% Confidence Interval')
    ax.fill_between(range(days_to_forecast), lower_bound_95, upper_bound_95, color='red', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(y=threshold1, color='purple', linestyle='--', alpha=0.5, label='Threshold 1', linewidth=2)
    ax.axhline(y=threshold2, color='orange', linestyle='--', alpha=0.5, label='Threshold 2', linewidth=2)
    ax.set_xlabel('Days', fontsize=15)
    ax.set_ylabel('Stock Price', fontsize=15)
    ax.set_title(f'Monte Carlo Confidence Cone for {ticker} using BSM Volatility', fontsize=20)
    ax.legend(fontsize=12)
    ax.grid(False)

    for day in range(0, days_to_forecast, 40): #Price estimation every 40 days:
        ax.annotate(f'{lower_bound_68[day]:.2f}', xy=(day, lower_bound_68[day]), xycoords='data', #MOre plot stuff
                    xytext=(-50, 0), textcoords='offset points', color='green', fontsize=10)
        ax.annotate(f'{upper_bound_68[day]:.2f}', xy=(day, upper_bound_68[day]), xycoords='data',
                    xytext=(-50, 0), textcoords='offset points', color='green', fontsize=10)
        ax.annotate(f'{lower_bound_95[day]:.2f}', xy=(day, lower_bound_95[day]), xycoords='data',
                    xytext=(-50, 0), textcoords='offset points', color='red', fontsize=10)
        ax.annotate(f'{upper_bound_95[day]:.2f}', xy=(day, upper_bound_95[day]), xycoords='data',
                    xytext=(-50, 0), textcoords='offset points', color='red', fontsize=10)

    above_threshold1_prob = (simulated_prices_bsm[-1] > threshold1).sum() / num_simulations #Probabilty of  staying in certin regions in accordance to our defined thresholds
    below_threshold2_prob = (simulated_prices_bsm[-1] < threshold2).sum() / num_simulations
    between_thresholds_prob = 1 - above_threshold1_prob - below_threshold2_prob

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) #Our box displaying those probabilities
    ax.text(0.02, 0.05, f'P(>{threshold1:.2f}): {above_threshold1_prob:.2%}\n'
                        f'P(<{threshold2:.2f}): {below_threshold2_prob:.2%}\n'
                        f'P({threshold2:.2f} - {threshold1:.2f}): {between_thresholds_prob:.2%}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)

    ax.annotate(f'{median_price_path[-1]:.2f}', xy=(days_to_forecast - 1, median_price_path[-1]), xycoords='data', #More plots
                xytext=(15, 0), textcoords='offset points', color='blue', fontsize=12)
    ax.annotate(f'{lower_bound_68[-1]:.2f}', xy=(days_to_forecast - 1, lower_bound_68[-1]), xycoords='data',
                xytext=(-15, 0), textcoords='offset points', color='green', fontsize=12)
    ax.annotate(f'{upper_bound_68[-1]:.2f}', xy=(days_to_forecast - 1, upper_bound_68[-1]), xycoords='data',
                xytext=(-15, 0), textcoords='offset points', color='green', fontsize=12)
    ax.annotate(f'{lower_bound_95[-1]:.2f}', xy=(days_to_forecast - 1, lower_bound_95[-1]), xycoords='data',
                xytext=(-15, 0), textcoords='offset points', color='red', fontsize=12)
    ax.annotate(f'{upper_bound_95[-1]:.2f}', xy=(days_to_forecast - 1, upper_bound_95[-1]), xycoords='data',
                xytext=(-15, 0), textcoords='offset points', color='red', fontsize=12)
    plt.show()

def run_simulation():
    check_for_updates()  # Check for updates before running the simulation
    global ticker
    global fig
    ticker = ticker_entry.get()
    prediction_days = int(prediction_days_entry.get()) 
    number_simulations = int(number_simulations_entry.get()) 
    BSM = float(BSM_entry.get()) 
    threshold1 = float(threshold1_entry.get()) if threshold1_entry.get() else None
    threshold2 = float(threshold2_entry.get()) if threshold2_entry.get() else None

    fig = monte_carlo(ticker, prediction_days, number_simulations, BSM, threshold1, threshold2)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

def tutorial():
    tutorial_window = Toplevel(window)
    tutorial_window.title('Monte Carlo Simulation Tutorial')
    tutorial_window.geometry('900x700')
    ttk.Label(tutorial_window, text=f'Monte Carlo Simulation Tutorial\n\n 1) Enter a ticker (Find on Yahoo finance)\n\n 2) Enter the amount of days you want to project the share price\n Enter the amount of times you want to run the simulation (recommend 500)\n\n3) You must then enter the BSM.\n This is the implied volatility you are assuming\nYou should try different volatilities between 0 and 1 and see how the simulation reacts\nYou can find the actual implied volatility of your stock by looking for the Options chain on Yahoo Finance or Bloomberg\n\n4) Once you have pressed run you will see the simulation graphic, you can still update all of the inputs and view the changes\n\n\n\nIf you have any questions please contact Mark Henry', foreground='black').grid(row=2, column=0, sticky='w')

def save_figure():
    global fig  # Access the global fig variable
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        fig.savefig(file_path)

window = tk.Tk()
window.title("Monte Carlo Simulation")

frame = ttk.Frame(window, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

tutorial_button = ttk.Button(frame, text="Read tutorial", command=tutorial)
tutorial_button.grid(row=0, column=2, columnspan=1)

save_button = ttk.Button(frame, text="Save Figure", command=save_figure)
save_button.grid(row=7, column=0, columnspan=1)

update_button = ttk.Button(frame, text="Check for Updates", command=check_for_updates)
update_button.grid(row=8, column=0, columnspan=2)

ttk.Label(frame, text="Ticker (Required):").grid(row=0, column=0, sticky=tk.W)
ticker_entry = ttk.Entry(frame)
ticker_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Prediction Days (Required):").grid(row=1, column=0, sticky=tk.W)
prediction_days_entry = ttk.Entry(frame)
prediction_days_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Number of Simulations (Required):").grid(row=2, column=0, sticky=tk.W)
number_simulations_entry = ttk.Entry(frame)
number_simulations_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="BSM (Required):").grid(row=3, column=0, sticky=tk.W)
BSM_entry = ttk.Entry(frame)
BSM_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Threshold 1:").grid(row=4, column=0, sticky=tk.W)
threshold1_entry = ttk.Entry(frame)
threshold1_entry.grid(row=4, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Threshold 2:").grid(row=5, column=0, sticky=tk.W)
threshold2_entry = ttk.Entry(frame)
threshold2_entry.grid(row=5, column=1, sticky=(tk.W, tk.E))

run_button = ttk.Button(frame, text="Run Simulation", command=run_simulation)
run_button.grid(row=6, column=0, columnspan=2)

window.mainloop()