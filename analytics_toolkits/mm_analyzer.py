import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import json

## For problem MM
def mm_check_requirements(row, mm_config):
    #print(row)
    #print(row["size"], row["mid"])
    required_size = mm_config[row["instrument"]]["requirement_size"]
    required_spread = mm_config[row["instrument"]]["requirement_spread"]
    current_spread = abs(row["mid"] - row["price"])*2 
    current_size = row["size"]
    row["is_min_size"] = 'N'
    row["is_max_spread"] = 'N'
    
    if current_size >= required_size:
        row["size_requirement_met"] = 'Y'
        if current_size == required_size: row["is_min_size"] = 'Y'
        
    else:
        row["size_requirement_met"] = 'N'
    
    if current_spread <= required_spread:
        row["spread_requirement_met"] = 'Y'
        if current_spread == required_spread: row["is_max_spread"] = 'Y' 
    else:
        row["spread_requirement_met"] = 'N'
    row["size_spread_requirement_met"] = 'Y' if row['size_requirement_met'] == 'Y' and row['spread_requirement_met'] == 'Y' else 'N'
    return row

def compute_ttl_time_meet_requirement(valid_time_interval):
    """ 
    compute total time for which trader meets requirement, O(N) 
    @params: 
        valid_time_interval: list of tuple in the form:[((start_ts, end_ts), duration), ...]
    @return:
        float, total time for which trader meets requirement
    """
    res = 0
    if len(valid_time_interval)>=1:
        res += valid_time_interval[0][1]
        last_end = valid_time_interval[0][0][1]
        for time_interval, dur in valid_time_interval[1:]:
            cur_start, cur_end = time_interval
            #print(last_end, cur_start, cur_end)
            new_dur = 0
            if cur_end <= last_end:
                pass
            else:
                if cur_start >= last_end:
                    new_dur = dur
                else:
                    new_dur = (cur_end - last_end).seconds
            res+=new_dur
            last_end = max(cur_end, last_end)
    return res
# For Problem Markouts
def read_n_process_markouts(prob2_file_config):
    res = {}
    instruments = prob2_file_config["tickers"]
    for instrument in instruments:
        instrument_config = json.load(open(os.path.join(prob2_file_config["root"], f"{instrument}.json")))
        # read raw data
        md = pd.read_csv(os.path.join(prob2_file_config["root"], f"{instrument}_md.csv"))
        trades = pd.read_csv(os.path.join(prob2_file_config["root"], f"{instrument}_trades.csv"))
        # process raw data
        md, trades = process_markouts_data(md, trades, instrument_config)
        res[instrument] = {
            "md": md,
            "trades": trades,
            "analytics": {}
        }
    return res

def process_markouts_data(md,trades, instrument_config):
    md = md.sort_values("ts_ms")
    trades = trades.sort_values("ts_ms")
    md["date"] = pd.to_datetime(md["ts_ms"], unit='ms')
    md["spread"] = md["ask"] - md["bid"]
    trades["date"] = pd.to_datetime(trades["ts_ms"], unit='ms')
    # filter out rows not match instrument config
    trades = trades.query(f"lhs_ccy == '{instrument_config['lhs_ccy']}' & rhs_ccy == '{instrument_config['rhs_ccy']}'").reset_index(drop=True)
    trades["dollar_volume"] = trades["size"] * trades["px"]
    return md, trades

def analyze_orders(markouts_data, start_datetime=None, end_datetime=None, order_cnt_freq="1D", plot_freq="1H", plots_dir="./plots",figsize=(20,8)):
    markouts = markouts_data.copy()
    for idx, (instrument, info) in enumerate(markouts.items()):
        md, trades = info["md"], info["trades"]
        order_cnt = get_order_cnt(trades, freq=order_cnt_freq)
        # generate order and trade plots
        plots = generate_order_plots(instrument, md, trades, start_datetime=start_datetime, end_datetime=end_datetime, 
                                     freq=plot_freq, plots_dir=plots_dir,figsize=figsize)
        markouts[instrument]["analytics"] = {
            "tbl_order_cnt": order_cnt,
            "plots": plots
        }
    return markouts

def generate_order_plots(instrument, md, trades, start_datetime=None, end_datetime=None, freq="1H",plots_dir="./plots",figsize=(20,8)):
    """
    @params:
        instrument: ticker, string
        md: top-of-book data, pd.DataFrame
        trades: trade data, pd.DataFrame
        start_datetime: starting time in interest, string
        end_datetime: ending time in interest, string
    """
    assert "date" in md.columns and "date" in trades.columns, "wrong input md and trades data, please run process_markouts_data() in advance"
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    start_time = md["date"][0]
    end_time = md["date"][len(md)-1]
    if start_datetime is not None:
        start_time = datetime.strptime(start_datetime,"%Y-%m-%d %H:%M:%S")
        md = md[md["date"]>start_datetime]
        trades = trades[trades["date"]>start_datetime]
    if end_datetime is not None:
        end_time = datetime.strptime(end_datetime,"%Y-%m-%d %H:%M:%S")
        md = md[md["date"]<end_datetime]
        trades = trades[trades["date"]<end_datetime]
    with_both_times = (start_datetime is not None and end_datetime is not None)
    title_tag1 = '' if start_datetime is None else start_datetime
    title_tag2 = '' if end_datetime is None else end_datetime

    # market data plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax11 = ax1.twinx()
    lns1 = ax1.plot(md["date"], md["bid"], label="bid", linewidth=0.5)
    lns2 = ax1.plot(md["date"], md["ask"], label="ask", linewidth=0.5)
    lns3 = []
    if (end_time - start_time).seconds > 43200:
        lns3 = ax11.plot(md["date"], md["spread"], 'r-.', label="spread")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    title = f"{instrument} - Top-of-book market data" if not (title_tag1 or title_tag2) else f"{instrument} - Top-of-book market data {title_tag1} - {title_tag2}"
    ax1.legend(lns, labs, loc=0)
    ax1.set_title(title)
    plt.savefig(f"{plots_dir}/{title}.pdf", dpi=1200)
    plt.close()
    # trade data plot
    ## trade volume per interval, e.g., 1 hour
    tmp = trades[["date", "side", "size"]]
    tmp.set_index("date", inplace=True)
    trade_volume = tmp.groupby("side").resample(freq).sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=figsize)
    pd.pivot(trade_volume[["date", "side", "size"]], index="date", values="size", columns="side").plot(ax=ax2)
    title = f"{instrument} - Trading Volume" if not (title_tag1 or title_tag2) else f"{instrument} - Trading Volume {title_tag1} - {title_tag2}"
    ax2.set_title(title)
    plt.savefig(f"{plots_dir}/{title}.pdf", dpi=1200)
    plt.close()

    ## dollar volume per interval, e.g., 1 hour
    tmp = trades[["date", "side", "dollar_volume"]]
    tmp.set_index("date", inplace=True)
    dollar_volume = tmp.groupby("side").resample(freq).sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=figsize)
    pd.pivot(dollar_volume[["date", "side", "dollar_volume"]], index="date", values="dollar_volume", columns="side").plot(ax=ax3)
    title = f"{instrument} - Dollar Volume" if not (title_tag1 or title_tag2) else f"{instrument} - Dollar Volume {title_tag1} - {title_tag2}"
    ax3.set_title(title)
    plt.savefig(f"{plots_dir}/{title}.pdf", dpi=1200)
    plt.close()
    figs = [fig1, fig2, fig3]
    return figs

def get_order_cnt(trades, freq="1D"):
    tmp = trades[["date", "side", "size"]].copy()
    tmp.set_index("date", inplace=True)
    daily_order_cnt = tmp.groupby("side").resample(freq)[["size"]].count().reset_index()
    daily_order_cnt.rename(columns={"size":"order_count"}, inplace=True)
    return daily_order_cnt

def compute_markout_pnl(row):
    mid_px, trade_side, trade_px, trade_size = row["mid"], row["side"],row["px"], row["size"]
    if trade_side == 'B':
        pnl = (mid_px - trade_px)*trade_size
    elif trade_side == 'S':
        pnl = (trade_px - mid_px)*trade_size
    return pnl
   
def generate_agg_markouts_plot(instrument, agg_markouts_df, horizon_ticks, figsize=(20,8), plots_dir="./plots"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(range(0, len(horizon_ticks),10))
    ax.set_xticklabels(horizon_ticks[::10],rotation = 45, ha="right")
    ax.tick_params(axis='x', which='major', labelsize=5)
    ax.tick_params(axis='x', which='minor', labelsize=5)
    ax.plot(agg_markouts_df["margins"])
    title = f"{instrument} - Aggregate Mark-out"
    ax.set_title(title)
    plt.savefig(f"{plots_dir}/{title}.pdf", dpi=1200)
    plt.close()
    return fig

def analyze_markouts(markouts_data, horizon_ticks, figsize=(20,8), plots_dir="./plots"):
    print("Computing Mark-outs")
    for idx, (instrument, info) in enumerate(markouts_data.items()):
        print(instrument)
        md, trades = info["md"], info["trades"]
        markouts_pnl = {}
        pnls = []
        margins = []
        ttl_trade_size = trades["size"].sum()
        pbar = tqdm(total=len(horizon_ticks))
        for i, trade_interval in enumerate(horizon_ticks):
            #print(trade_interval)
            trades["ts_ms_markouts"] = trades["ts_ms"]+trade_interval
            trades_wt_markouts = pd.merge_asof(trades, md, left_on="ts_ms_markouts", right_on="ts_ms", direction="backward", suffixes=("_trade","_order"))
            trades_wt_markouts["mid"] = (trades_wt_markouts["bid"] + trades_wt_markouts["ask"])/2
            trades_wt_markouts["pnl"] = trades_wt_markouts.apply(lambda row: compute_markout_pnl(row), axis=1)
            markouts_pnl[trade_interval] = trades_wt_markouts[["trade_id","ts_ms_order","mid","pnl"]]
            ttl_pnl = trades_wt_markouts["pnl"].sum()
            pnls.append(ttl_pnl)
            margins.append(ttl_pnl/ttl_trade_size*10000)
            pbar.update(1)
        pbar.close()
        markout_res = pd.DataFrame({
            "horizon_ticks": horizon_ticks,
            "margins": margins
        })
        analytics = markouts_data[instrument]["analytics"]
        analytics.update({
            "markouts_pnl_per_interval": markouts_pnl,
            "markout_agg_result": markout_res
        })
        markouts_plot = generate_agg_markouts_plot(instrument, markout_res, horizon_ticks, figsize=figsize,plots_dir=plots_dir)
        analytics["plots"] = analytics.get("plots",[])+[markouts_plot]