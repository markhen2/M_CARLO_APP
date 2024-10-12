import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import customtkinter
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

GITHUB_REPO = "markhen2/Stock-review-app.git"
CURRENT_VERSION = "1.0.0"

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
    with open("update.zip", "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile("update.zip", "r") as zip_ref:
        zip_ref.extractall("update")
    os.remove("update.zip")
    apply_update("update")
    shutil.rmtree("update")

def apply_update(update_dir):
    for item in os.listdir(update_dir):
        s = os.path.join(update_dir, item)
        d = os.path.join(os.getcwd(), item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("Update applied. Please restart the application.")

def monte_carlo(ticker, prediction_days, number_simulations, BSM=0.26, threshold1=None, threshold2=None):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    df = yf.download(ticker, start=start_date, end=end_date)
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    days_to_forecast = prediction_days
    num_simulations = number_simulations
    dt = 1
    if threshold2 is None:
        threshold2 = df['Adj Close'][-1] * 1.9
    if threshold1 is None:
        threshold1 = df['Adj Close'][-1] * 0.5

    volatility_bsm = BSM
    print("BSM Implied Volatility: ", volatility_bsm)

    def run_simulation(volatility, dt, annualized=False):
        simulated_prices = np.zeros((days_to_forecast, num_simulations))
        simulated_prices[0] = df['Close'][-1]

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